import numpy as np
import pytest
import torch
from types import SimpleNamespace
from config import Config
from sampling import StepDistanceSampler, pca_two
from test_compact import _stub_gpu_dependencies


def test_metrics_have_distinct_geometry():
    x = np.array([[1., 0.], [10., 0.], [0., 1.]], dtype=np.float32)
    e = StepDistanceSampler(x, 'euclidean', seed=42)
    c = StepDistanceSampler(x, 'cosine', seed=42)
    for sampler in (e, c):
        sampler.previous = [0]
        sampler.available[0] = False
    assert e.sample(1) == [1]  # far in L2 length
    assert c.sample(1) == [2]  # far in angle


def test_sampler_unique_and_deterministic():
    x = np.random.default_rng(3).normal(size=(103, 256)).astype(np.float32)
    for method in ('euclidean', 'cosine', 'pca_knn'):
        a = StepDistanceSampler(x, method, seed=5, k=2)
        b = StepDistanceSampler(x, method, seed=5, k=2)
        chosen = []
        for _ in range(25):
            qa, qb = a.sample(4), b.sample(4)
            assert qa == qb
            chosen.extend(qa)
        assert len(chosen) == len(set(chosen)) == 100
        if method == 'pca_knn':
            assert len(a.explained_variance_ratio) == 2
            assert 0 < sum(a.explained_variance_ratio) <= 1.0001


def test_pca_does_not_change_training_rng():
    x = np.random.default_rng(4).normal(size=(32,8)).astype('float32')
    before = torch.random.get_rng_state().clone()
    _ = pca_two(x, seed=17)
    torch.testing.assert_close(before, torch.random.get_rng_state())


def test_policy_backbone_forward_without_head_or_grad(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    from distributed import PolicyWorker
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2, 3, 4, 5]
    class Backbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, return_dict):
            assert torch.is_grad_enabled() is False
            assert use_cache is False
            return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids % 16, 16).float())
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.config = SimpleNamespace(hidden_size=16)
    obj = SimpleNamespace(
        config=SimpleNamespace(sampling_max_length=4, sampling_projection_dim=3),
        tokenizer=Tokenizer(), model=Model(),
        rollout=SimpleNamespace(device=torch.device('cpu')),
    )
    rows = [{'messages':[{'role':'user','content':'a'}]}]

    # The embedding method may only be used while Task's GPU projector exists.
    with pytest.raises(RuntimeError, match='begin_sampling_projection'):
        PolicyWorker.embed_sampling_batch(obj, rows, seed=42)
    rng_before = torch.random.get_rng_state().clone()
    PolicyWorker.begin_sampling_projection(obj, seed=42)
    projector = obj._sampling_projector
    torch.testing.assert_close(rng_before, torch.random.get_rng_state())
    assert projector.device == obj.rollout.device
    assert projector.shape == (3, 16) and not projector.requires_grad
    out1 = PolicyWorker.embed_sampling_batch(obj, rows, seed=42)
    out2 = PolicyWorker.embed_sampling_batch(obj, rows, seed=42)
    assert obj._sampling_projector is projector
    assert obj.model.training is True
    assert len(out1) == 1 and len(out1[0]) == 3 and out1 == out2

    # The GPU projector vanishes before GRPO; a new Task recreates it.
    PolicyWorker.end_sampling_projection(obj)
    assert not hasattr(obj, '_sampling_projector')
    with pytest.raises(RuntimeError, match='begin_sampling_projection'):
        PolicyWorker.embed_sampling_batch(obj, rows, seed=42)
    PolicyWorker.begin_sampling_projection(obj, seed=42)
    again = PolicyWorker.embed_sampling_batch(obj, rows, seed=42)
    assert out1 == again
    PolicyWorker.end_sampling_projection(obj)
    PolicyWorker.begin_sampling_projection(obj, seed=43)
    different = PolicyWorker.embed_sampling_batch(obj, rows, seed=43)
    assert out1 != different
    PolicyWorker.end_sampling_projection(obj)


def test_embedding_cleanup_on_failure(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    import sampling

    events = []
    class RPC:
        def __init__(self, fn): self.fn = fn
        def remote(self, *args): return self.fn(*args)
    class Worker:
        def __init__(self, rank):
            self.rank = rank
            self.begin_sampling_projection = RPC(lambda seed: events.append((rank, 'start')))
            self.end_sampling_projection = RPC(lambda: events.append((rank, 'end')))
            def embed(rows, seed):
                events.append((rank, 'embed'))
                raise RuntimeError('synthetic embedding failure')
            self.embed_sampling_batch = RPC(embed)

    cfg = SimpleNamespace(
        parallel_mode='tp', sampling_embed_batch_size=1,
        sampling_projector_seed=2026, sampling_max_length=256,
        sampling_projection_dim=256, progress_bar=False,
        ray_dist_timeout_s=60,
    )
    rows = [{'messages': [{'role':'user', 'content':'Q'}]}]
    with pytest.raises(RuntimeError, match='synthetic embedding failure'):
        sampling.embed_entire_task([Worker(0), Worker(1)], cfg, rows, 'mathematics')
    assert events.count((0, 'start')) == events.count((1, 'start')) == 1
    assert events.count((0, 'end')) == events.count((1, 'end')) == 1


def test_variant_isolated_and_original_unchanged(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    from continual_train.train import build_track_config, SIGNATURE_FIELDS
    base=Config()
    base.parallel_mode='tp'
    base.hf_model_repo='changgyu/recursive-rl-continual-model-tp'
    base.hf_continual_repo='changgyu/recursive-rl-continual-data'
    base.eval_restore_revision='a'*40
    base.eval_restore_counts='{}'
    original=build_track_config(base, 'grpo')
    assert original.hf_model_repo == base.hf_model_repo
    assert 'train_sampler' not in SIGNATURE_FIELDS  # preserves prior progress.json format
    for method in ('euclidean','cosine','pca_knn'):
        base.train_sampler=method
        cfg=build_track_config(base,'grpo')
        assert cfg.hf_model_repo.endswith('-'+method.replace('_','-'))
        assert cfg.baseline_source_repo==base.hf_model_repo
        assert cfg.continual_checkpoint_dir!=original.continual_checkpoint_dir
        assert cfg.eval_restore_counts is None
        assert cfg.train_batch_size==original.train_batch_size
        assert cfg.grpo_group_size==original.grpo_group_size


def test_new_policy_is_exact_grpo_subclass():
    from policy import GRPOPolicy, DistanceSamplingGRPOPolicy
    assert issubclass(DistanceSamplingGRPOPolicy, GRPOPolicy)
    assert DistanceSamplingGRPOPolicy.loss is GRPOPolicy.loss
    assert DistanceSamplingGRPOPolicy.advantages is GRPOPolicy.advantages
    assert DistanceSamplingGRPOPolicy.train_batch is GRPOPolicy.train_batch


def test_sampling_integrates_with_existing_grpo_step(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    import sampling
    from continual_train import train as trainer_module
    from continual_train.train import ContinualTrainer
    embeddings=np.random.default_rng(7).normal(size=(24, 32)).astype('float32')
    monkeypatch.setattr(sampling, 'embed_entire_task', lambda w,c,r,d: embeddings)
    dispatched=[]
    class RPC:
        def __init__(self, fn): self.fn=fn
        def remote(self,*args): return self.fn(*args)
    class Worker:
        def __init__(self):
            self.cleanup_gpu=RPC(lambda: None)
            self.rollout_batch=RPC(lambda d,rows,micro,acc: dispatched.append((d, rows, micro,acc)))
    monkeypatch.setattr(trainer_module, 'await_workers_progress',
                        lambda cfg, requests, stage, **kw: [{} for _ in requests])
    cfg=Config()
    cfg.parallel_mode='tp'; cfg.train_sampler='cosine'
    cfg.train_batch_size=4;cfg.train_finance_batch_size=2
    cfg.train_finance_gradient_accumulation=2
    cfg.train_max_steps=1;cfg.trace_mode=False;cfg.progress_bar=False
    data=[{'messages':[{'role':'user', 'content':f'q{i}'}], 'answer':'0'} for i in range(24)]
    workers=[Worker(), Worker()]
    steps=ContinualTrainer(cfg).train_domain(workers, 'mathematics', data, task_id=1)
    assert steps==1 and len(dispatched)==2
    assert all(len(entry[1])==4 and entry[2:]==(0,1) for entry in dispatched)
    assert dispatched[0][1] == dispatched[1][1]   # same TP inputs on both ranks
    dispatched.clear()
    steps=ContinualTrainer(cfg).train_domain(workers, 'finance', data, task_id=4)
    assert steps==1 and len(dispatched)==4
    assert [e[2:] for e in dispatched] == [(0,2),(0,2),(1,2),(1,2)]
    assert all(len(entry[1])==2 for entry in dispatched)
    assert dispatched[0][1]==dispatched[1][1]
    assert dispatched[2][1]==dispatched[3][1]
