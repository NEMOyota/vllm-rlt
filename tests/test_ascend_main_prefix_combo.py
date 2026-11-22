"""Exercise the experiment's gates with the mainline host algorithms on CPU."""

from weakref import WeakKeyDictionary

import pytest
import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks.ascend_prefix_combo_checks import PrefillAudit, page_oracle
from tests.helpers import tiny_ouro_config
from tests.test_device import npu_model_stub
from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine import stage_engine
from vllm_rlt.kernels.ascend_attention import AscendAttention
from vllm_rlt.models import OuroForCausalLM


def cpu_attention():
    return AttentionBackend(
        AscendAttention(),
        BackendCapabilities(packed_prefill=True),
        {"backend": "ascend_eager", "selected_backend": "ascend_eager"},
    )


def test_experiment_config_reaches_npu_packed_backend_selection(monkeypatch):
    class BackendSelected(Exception):
        pass

    def select(name, device, dtype, *args, prefer_packed_prefill, **kwargs):
        assert (name, device.type, dtype) == ("ascend_eager", "npu", torch.bfloat16)
        assert prefer_packed_prefill
        raise BackendSelected

    monkeypatch.setattr(stage_engine, "create_backend", select)
    with pytest.raises(BackendSelected):
        combo.open_engine(npu_model_stub(), combo.CASES["c4_h2"], "reuse")


def test_shared_page_oracle_on_cpu():
    cache = KVCacheManager(
        2,
        2,
        64,
        32,
        16,
        4,
        dtype=torch.bfloat16,
        attention=cpu_attention(),
        enable_prefix_caching=True,
    )
    cache._prepared_coverage = WeakKeyDictionary()
    page_oracle(cache)


def test_three_arms_through_cpu_runner(monkeypatch):
    # Enable the real packed host descriptors and range algorithm on CPU.
    # The server entry point itself always requires torch-npu.
    build = combo.open_engine
    monkeypatch.setattr(stage_engine, "packed_prefill_eligible", lambda *a, **kw: True)
    monkeypatch.setattr(stage_engine, "create_backend", lambda *a, **kw: cpu_attention())

    def host_engine(*args):
        engine = build(*args)
        engine.cache_manager._prepared_coverage = WeakKeyDictionary()
        return engine

    monkeypatch.setattr(combo, "open_engine", host_engine)
    torch.manual_seed(168)
    model = OuroForCausalLM(tiny_ouro_config()).to(torch.bfloat16).eval()
    case = combo.Case(16, 2, 4, 2)
    donor, miss, targets = combo.prompts(case, model.config.vocab_size)
    for arm in combo.ARMS:
        result = combo.run_arm(model, case, arm, miss if arm == "miss" else donor, targets)
        assert result["hit_tokens"] == case.starts(arm)
        assert result["audit"]["packed_descriptors"] == 4
        assert result["audit"]["range_updates"] == 4 * 4 * 2
        if arm == "reuse":
            assert result["audit"]["explicit_masks_checked"] > 0


def test_cached_suffix_rejects_top_left_causal_mask():
    audit = PrefillAudit(None, {"target-0": 16}, 18)
    with pytest.raises(AssertionError, match="mask"):
        audit.check_mask(2, 18, None, True)
    with pytest.raises(AssertionError, match="mask"):
        audit.check_mask(2, 18, torch.ones(2, 18, dtype=torch.bool), False)


def test_token_flip_is_reported_per_request_at_one_based_position():
    ref = {"requests": {"target-2": {"tokens": [302, 30]}}}
    other = {"requests": {"target-2": {"tokens": [302, 93]}}}
    assert combo.token_diffs(ref, other, "off_vs_reuse") == [
        dict(
            comparison="off_vs_reuse",
            request="target-2",
            first_diff_1based=2,
            expected=30,
            actual=93,
        )
    ]
