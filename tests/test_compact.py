"""Fast CPU-only invariants; real two-GPU NCCL validation must run on RunPod."""
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from data_load import DataLoader, PromptFormatter
from policy import GRPOPolicy, reward_breakdown, evaluation_correctness, SelfImprovePolicy
from continual_train.storage import (atomic_write_json, evaluation_fingerprint,
                                      validate_baseline, restore_from_logged_counts)
from visualization import render_saved


def _stub_gpu_dependencies(monkeypatch):
    """Allow import-time verification of orchestration without real Ray/HF model."""
    ray = types.ModuleType('ray')
    ray.__path__ = []
    ray.remote = lambda **kwargs: lambda cls: cls
    ray.get = lambda values, **kwargs: values
    ray.wait = lambda values, **kwargs: (values[:1], values[1:])
    util = types.ModuleType('ray.util')
    util.__path__ = []
    util.placement_group = types.ModuleType('ray.util.placement_group')
    util.placement_group.placement_group = lambda *args, **kwargs: None
    util.placement_group.remove_placement_group = lambda *args, **kwargs: None
    util.scheduling_strategies = types.ModuleType('ray.util.scheduling_strategies')
    util.scheduling_strategies.PlacementGroupSchedulingStrategy = lambda **kw: kw
    ray.util = util
    for key, obj in {'ray': ray, 'ray.util': util,
                     'ray.util.placement_group': util.placement_group,
                     'ray.util.scheduling_strategies': util.scheduling_strategies}.items():
        monkeypatch.setitem(sys.modules, key, obj)
    hf = types.ModuleType('transformers')
    hf.AutoConfig = type('AutoConfig', (), {})
    hf.AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
    hf.AutoTokenizer = type('AutoTokenizer', (), {})
    monkeypatch.setitem(sys.modules, 'transformers', hf)


def test_import_dependency_graph(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    for name in ['rollout', 'distributed', 'continual_train.evaluate', 'continual_train.train']:
        importlib.import_module(name)


def test_batch_router_parallel_semantics(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    from distributed import BatchRouter
    for mode, expected in [('tp', 4), ('fsdp', 8)]:
        router = BatchRouter(SimpleNamespace(parallel_mode=mode), 2, 4)
        assert router.global_size == expected
        batches = router.split(list(range(expected)))
        assert batches == ([[0,1,2,3]] * 2 if mode == 'tp'
                           else [[0,1,2,3],[4,5,6,7]])
        assert router.unique(['gpu0','gpu1']) == (['gpu0'] if mode == 'tp' else ['gpu0','gpu1'])
        with pytest.raises(ValueError, match='Expected'):
            router.split([])


def test_policy_rewards_and_objective_unchanged():
    cfg = Config()
    examples = [
        ('Final Answer: 42\n', {'answer':'42'}, 'eos', 5.0),
        ('Final Answer: 2.0', {'answer':'2'}, None, 5.0),
        ('Final Answer: A\nNEXT QUESTION', {'answer':'A','answer_letter':'A'}, 'max_tokens', 4.098361),
        ('The result is 8', {'answer':'8'}, 'eos', 0.0),
    ]
    for answer, row, reason, expected in examples:
        actual = reward_breakdown('mathematics', answer, row, finish_reason=reason)
        assert actual['total_reward'] == expected
        assert evaluation_correctness('mathematics', answer, row, finish_reason=reason) == actual['correctness_reward']
    grpo = GRPOPolicy(cfg)
    torch.manual_seed(42)
    rewards, logps = torch.randn(8), torch.randn(8,5)
    advantage = grpo.advantages(rewards,4)
    grouped = rewards.reshape(-1,4)
    torch.testing.assert_close(advantage, ((grouped-grouped.mean(1,keepdim=True)) /
                                          (grouped.std(1,keepdim=True,unbiased=False)+1e-4)).flatten())
    rollout = {'old_logps': torch.randn(8,5), 'advantages': advantage,
               'completion_mask': torch.ones(8,5,dtype=torch.bool)}
    ratio = (logps - rollout['old_logps']).clamp(-20,20).exp()
    clipped = ratio.clamp(1-cfg.grpo_clip_range,1+cfg.grpo_clip_range)
    expected_loss = -torch.minimum(ratio * advantage[:,None], clipped * advantage[:,None]).mean()
    torch.testing.assert_close(grpo.loss(logps,rollout),expected_loss)
    assert SelfImprovePolicy.IMPLEMENTED is False


def test_prompt_format_and_data_loader():
    row = {'question':'What is 2+2?', 'answer':'4', 'answer_letter':'', 'choices':{}, 'reference':''}
    messages = PromptFormatter().build_messages(row, 'mathematics')
    assert messages[0]['role'] == 'system'
    assert 'Final Answer:' in messages[1]['content']
    assert DataLoader._record('What is 2+2?', '4')['answer'] == '4'


def test_atomic_storage_and_baseline_checks(tmp_path):
    path = tmp_path/'sub'/'progress.json'
    atomic_write_json(path, {'a':1}, indent=2)
    atomic_write_json(path, {'a':2}, indent=2)
    assert json.loads(path.read_text()) == {'a':2}
    eval_data = {'mathematics':[{'messages':[{'role':'user','content':'one'}], 'answer':'4',
                                 'answer_letter':'', 'reference':''}]}
    cfg = SimpleNamespace(baseline_restore_counts='{"mathematics":[1,1]}',
                          train_domains=['mathematics'])
    score = restore_from_logged_counts(cfg, eval_data)
    validate_baseline(score, cfg.train_domains, {'mathematics':1})
    assert score['mathematics']['accuracy'] == 1
    assert evaluation_fingerprint(eval_data) != evaluation_fingerprint({'mathematics':[{
        'messages':[{'role':'user','content':'different'}],'answer':'4',
        'answer_letter':'','reference':''}]})


def test_matplotlib_saved_pca(tmp_path):
    payload = {'samples':[{'id':'a','domain':'mathematics'},{'id':'b','domain':'science'}],
               'frames':[{'stage':'before_training','points':[[0,0],[1,1]]},
                         {'stage':'after_mathematics','points':[[.2,.2],[1.1,1.2]]}]}
    (tmp_path/'frames.json').write_text(json.dumps(payload))
    files = render_saved(tmp_path)
    assert len(files) >= 3
    assert (tmp_path/'pca_before_training.png').is_file()
    assert (tmp_path/'pca_trajectory.png').is_file()
    assert (tmp_path/'pca_latest.png').is_file()


def test_strict_resume_signature_preserved(monkeypatch):
    _stub_gpu_dependencies(monkeypatch)
    from continual_train.train import SIGNATURE_FIELDS, build_track_config
    assert {'train_max_steps','train_lr','grpo_group_size', 'parallel_mode'} <= set(SIGNATURE_FIELDS)
    base = Config()
    cfg = build_track_config(base,'self_improve')
    assert cfg.continual_checkpoint_dir != base.continual_checkpoint_dir
    assert cfg.train_seed == base.train_seed
    assert cfg.train_domains == base.train_domains
