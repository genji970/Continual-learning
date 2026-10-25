"""Persistent storage and task-0 baseline compatibility.

HF private model repository is authoritative. Distributed checkpoints always
include model AND optimizer; restore uses immutable Hub revision round-trip.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

# Compatibility fields for reusing the existing, untrained baseline from another
# independent sampling experiment. (Missing in the uploaded merged storage.py.)
BASELINE_COMPAT_FIELDS = (
    "model_name", "model_revision", "model_dtype", "parallel_mode",
    "hf_continual_repo", "train_domains", "continual_eval_limit",
    "grpo_max_prompt_length", "grpo_max_completion_length",
)


def atomic_write_json(path: Path, value, *, indent: int | None = None) -> None:
    """Replace a JSON file only after the entire new content is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

class TrainingState(Stateful):
    """FSDP2 model/optimizer share one distributed checkpoint state."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model, optimizer = get_state_dict(self.model, self.optimizer)
        return {"model": model, "optimizer": optimizer}

    def load_state_dict(self, state):
        set_state_dict(self.model, self.optimizer,
                       model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])


def save_checkpoint(model, optimizer, path: str):
    # Invoked simultaneously by every NCCL rank.
    dcp.save({"training": TrainingState(model, optimizer)}, checkpoint_id=path)


def load_checkpoint(model, optimizer, path: str):
    dcp.load({"training": TrainingState(model, optimizer)}, checkpoint_id=path)

class PrivateTaskCheckpoints:
    def __init__(self, config):
        self.repo_id = config.resolve_hf_model_repo()
        self.token = os.getenv(config.hf_token_env) or None
        self.api = HfApi(token=self.token)
        # Existing public repositories MUST NOT be used for checkpoints.
        self.api.create_repo(
            repo_id=self.repo_id, repo_type="model", private=True,
            exist_ok=True, token=self.token,
        )
        self._ensure_private()

    def _ensure_private(self):
        info = self.api.repo_info(
            repo_id=self.repo_id, repo_type="model", token=self.token,
        )
        if info.private is not True:
            raise RuntimeError(
                f"Refusing checkpoint transfer: {self.repo_id} is NOT private. "
                "Use a different private model repository."
            )

    @staticmethod
    def _check_dcp_folder(folder: Path):
        if not (folder / ".metadata").is_file():
            raise FileNotFoundError(f"Distributed checkpoint metadata missing: {folder}")
        if not any(folder.rglob("*.distcp")):
            raise FileNotFoundError(f"Distributed checkpoint shards missing: {folder}")

    @staticmethod
    def _remote_path(name: str) -> str:
        p = PurePosixPath(name)
        if (len(p.parts) != 2 or p.parts[0] != "checkpoints"
                or not re.fullmatch(r"task_[0-9]{2}_[A-Za-z0-9_]+_[a-f0-9]{8}", p.parts[1])):
            raise ValueError(f"Invalid remote checkpoint path: {name!r}")
        return str(p)

    def load_progress(self):
        """Remote progress is authoritative; network/authentication failures are fatal."""
        self._ensure_private()
        exists = self.api.file_exists(
            repo_id=self.repo_id, filename="progress.json", repo_type="model",
            token=self.token,
        )
        if not exists:
            return None
        path = hf_hub_download(
            repo_id=self.repo_id, filename="progress.json", repo_type="model",
            token=self.token, force_download=True,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def publish_progress(self, folder: Path, progress: dict):
        """Publish progress directly to the private HF Model Repo, then verify.

        No project-local progress.json is required. The Hub is authoritative.
        The folder argument remains for compatibility with existing callers.
        """
        self._ensure_private()
        payload = (json.dumps(progress, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.api.upload_file(
            repo_id=self.repo_id, repo_type="model", token=self.token,
            path_or_fileobj=io.BytesIO(payload),
            path_in_repo="progress.json",
            commit_message="Update continual training progress",
        )
        verified = self.load_progress()
        if verified != progress:
            raise RuntimeError("HF progress.json verification failed after upload")
        print(f"[HF PROGRESS VERIFIED] {self.repo_id}/progress.json", flush=True)

    def upload_checkpoint(self, folder: Path) -> dict[str, str]:
        """Upload a complete DCP directory and verify all shard paths exist."""
        folder = Path(folder)
        self._check_dcp_folder(folder)
        self._ensure_private()
        destination = self._remote_path("checkpoints/" + folder.name)
        print(f"[HUB] Uploading private checkpoint: {self.repo_id}/{destination}", flush=True)
        commit = self.api.upload_folder(
            repo_id=self.repo_id, repo_type="model", token=self.token,
            folder_path=str(folder), path_in_repo=destination,
            commit_message=f"FSDP2 full model and optimizer: {folder.name}",
        )
        revision = getattr(commit, "oid", None)
        if not revision:
            raise RuntimeError("Hub did not return a checkpoint commit revision")
        self._ensure_private()
        found = set(self.api.list_repo_files(
            repo_id=self.repo_id, repo_type="model", revision=revision,
            token=self.token,
        ))
        expected = {
            f"{destination}/{p.relative_to(folder).as_posix()}"
            for p in folder.rglob("*") if p.is_file()
        }
        missing = expected - found
        if missing:
            raise RuntimeError(
                f"Hub checkpoint incomplete: {len(missing)} files missing "
                f"(example: {next(iter(sorted(missing)))})"
            )
        print(f"[HUB] Upload verified, revision={revision}", flush=True)
        return {"checkpoint": destination, "checkpoint_revision": revision}

    def verify_checkpoint_record(self, entry: dict):
        """Check a completed checkpoint is present without downloading all weights."""
        path = self._remote_path(entry["checkpoint"])
        revision = entry.get("checkpoint_revision")
        if not revision or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("Checkpoint record has no immutable Hub revision")
        self._ensure_private()
        files = set(self.api.list_repo_files(
            repo_id=self.repo_id, repo_type="model", revision=revision,
            token=self.token,
        ))
        if f"{path}/.metadata" not in files or not any(
            name.startswith(path + "/") and name.endswith(".distcp")
            for name in files
        ):
            raise RuntimeError(f"Hub checkpoint not complete: {path}@{revision}")

    def download_checkpoint(self, entry: dict, root: Path) -> str:
        """Download a pinned remote revision into a NEW local path for DCP load."""
        path = self._remote_path(entry["checkpoint"])
        revision = entry.get("checkpoint_revision")
        if not revision or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("Checkpoint has no immutable Hub revision")
        self._ensure_private()
        target = Path(root) / "hub_downloads" / PurePosixPath(path).name
        print(f"[HUB] Downloading private checkpoint: {path}@{revision}", flush=True)
        snapshot_download(
            repo_id=self.repo_id, repo_type="model", revision=revision,
            token=self.token, local_dir=str(target),
            allow_patterns=[f"{path}/*", f"{path}/**"],
        )
        folder = target / path
        self._check_dcp_folder(folder)
        remote = {
            f"{path}/{p.relative_to(folder).as_posix()}"
            for p in folder.rglob("*") if p.is_file()
        }
        listed = set(self.api.list_repo_files(
            repo_id=self.repo_id, repo_type="model", revision=revision,
            token=self.token,
        ))
        expected = {key for key in listed if key.startswith(path + "/")}
        if not expected or not expected.issubset(remote):
            raise RuntimeError("Downloaded checkpoint is missing remote files")
        self._ensure_private()
        print(f"[HUB] Download verified: {folder}", flush=True)
        return str(folder)

def evaluation_fingerprint(evaluation: dict[str, list[dict]]) -> str:
    """Identify ordered evaluation prompts and targets; ignore trace metadata."""
    digest = hashlib.sha256()
    for domain, rows in evaluation.items():
        digest.update(json.dumps([domain, len(rows)], ensure_ascii=False).encode())
        for row in rows:
            record = {name: row.get(name) for name in
                      ("messages", "answer", "answer_letter", "reference")}
            digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def validate_baseline(baseline: dict, expected_domains: list[str],
                      expected_sizes: dict[str, int]) -> None:
    if not isinstance(baseline, dict) or set(baseline) != set(expected_domains):
        raise ValueError("Saved baseline domains differ from current evaluation")
    for domain in expected_domains:
        result = baseline[domain]
        if not isinstance(result, dict):
            raise ValueError(f"Invalid saved baseline result for {domain}")
        total, correct, accuracy = (result.get(key) for key in ("total", "correct", "accuracy"))
        if (type(total) is not int or type(correct) is not int or
                total != expected_sizes[domain] or not 0 <= correct <= total or
                not isinstance(accuracy, (int, float)) or
                not math.isfinite(accuracy) or
                not math.isclose(accuracy, correct / total, abs_tol=1e-8)):
            raise ValueError(f"Saved baseline scores/counts invalid for {domain}")


def restore_from_logged_counts(config, evaluation: dict[str, list[dict]]) -> dict:
    """Restore only aggregate Task-0 scores recorded by a completed earlier run.

    The user must confirm that the previous run used the same model and eval data.
    It is impossible to verify this solely from printed aggregate counts.
    """
    try:
        raw = json.loads(config.baseline_restore_counts)
    except (TypeError, ValueError) as exc:
        raise ValueError("RL_BASELINE_RESTORE_COUNTS must be a JSON object") from exc
    if not isinstance(raw, dict) or set(raw) != set(evaluation):
        raise ValueError("Restore counts must contain exactly the configured eval domains")

    restored = {}
    for domain, records in evaluation.items():
        item = raw[domain]
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{domain}: expected [correct, total]")
        correct, total = item
        if type(correct) is not int or type(total) is not int:
            raise ValueError(f"{domain}: correct and total must be integers")
        restored[domain] = {
            "correct": correct, "total": total,
            "accuracy": correct / total if total > 0 else -1,
        }
    validate_baseline(
        restored, config.train_domains,
        {domain: len(rows) for domain, rows in evaluation.items()},
    )
    return restored


def _validate_compatibility(saved: dict, signature: dict, source: str,
                            *, external: bool) -> None:
    prior = saved.get("signature")
    if not isinstance(prior, dict):
        raise ValueError(f"{source} is missing progress signature")
    # The same track still uses the stricter existing resume-signature check.
    if external:
        mismatched = [key for key in BASELINE_COMPAT_FIELDS
                      if key not in prior or prior[key] != signature.get(key)]
        if mismatched:
            raise ValueError(f"{source}: incompatible baseline settings: {', '.join(mismatched)}")


def _read_other_repo(hub, repo_id: str) -> dict | None:
    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Invalid RL_BASELINE_SOURCE_REPO")
    info = hub.api.repo_info(repo_id=repo_id, repo_type="model", token=hub.token)
    if info.private is not True:
        raise RuntimeError("Baseline source repo must be private")
    if not hub.api.file_exists(repo_id=repo_id, filename="progress.json",
                               repo_type="model", token=hub.token):
        return None
    path = hf_hub_download(repo_id=repo_id, filename="progress.json",
                           repo_type="model", token=hub.token, force_download=True)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_baseline(config, progress: dict, hub, evaluation: dict,
                    signature: dict) -> tuple[dict | None, str | None, str]:
    """Return (baseline, fingerprint, reason).

    'auto': reuse if present, otherwise evaluate.
    'reuse': require existing compatible Hub baseline; never run eval as fallback.
    'evaluate': force a fresh baseline, only before training any task.
    """
    mode = config.baseline_mode
    fingerprint = evaluation_fingerprint(evaluation)
    sizes = {domain: len(rows) for domain, rows in evaluation.items()}

    if progress.get("history"):
        if mode in {"evaluate", "restore"}:
            raise ValueError("Cannot evaluate original base model after tasks have trained")
        existing = progress.get("baseline_accuracy")
        if config.benchmark_forward_transfer and existing is None:
            raise ValueError("Completed task history has no baseline to reuse")
        if existing is not None:
            validate_baseline(existing, config.train_domains, sizes)
            saved_fp = progress.get("baseline_fingerprint")
            if saved_fp and saved_fp != fingerprint:
                raise ValueError("Saved baseline evaluation-data fingerprint mismatch")
        return existing, fingerprint if existing else None, "resumed tasks"

    if not config.benchmark_forward_transfer:
        if mode != "auto":
            raise ValueError("Set RL_BENCHMARK_FORWARD_TRANSFER=true to use a baseline mode")
        return None, None, "baseline disabled"

    if mode == "evaluate":
        return None, None, "forced reevaluation"
    if mode == "restore":
        return None, None, "restore recorded baseline"

    candidates = [(hub.repo_id, progress, False)]
    if config.baseline_source_repo and config.baseline_source_repo != hub.repo_id:
        candidates.append((config.baseline_source_repo, None, True))

    for repo_id, saved, external in candidates:
        if external:
            saved = _read_other_repo(hub, repo_id)
        if not saved or not saved.get("baseline_accuracy"):
            continue
        _validate_compatibility(saved, signature, repo_id, external=external)
        prior_fp = saved.get("baseline_fingerprint")
        if prior_fp is not None and prior_fp != fingerprint:
            raise ValueError(f"{repo_id}: baseline evaluation-data fingerprint mismatch")
        validate_baseline(saved["baseline_accuracy"], config.train_domains, sizes)
        if prior_fp is None:
            print(f"[BASELINE WARNING] {repo_id}: legacy baseline without data fingerprint; "
                  "verified settings and counts only. Reuse assumes identical evaluation data.",
                  flush=True)
        return saved["baseline_accuracy"], fingerprint, f"Hugging Face {repo_id}"

    if mode == "reuse":
        raise FileNotFoundError(
            "RL_BASELINE_MODE=reuse, but no completed baseline_accuracy was found "
            "in this track's private progress.json or RL_BASELINE_SOURCE_REPO. "
            "An interrupted baseline evaluation cannot be reused."
        )
    return None, None, "no compatible baseline; evaluate"
