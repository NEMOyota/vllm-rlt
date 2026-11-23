"""Exercise the experiment's gates with the mainline host algorithms on CPU."""

from types import SimpleNamespace
from weakref import WeakKeyDictionary

import pytest
import torch
from torch import nn

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_linear as linear
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


@pytest.fixture
def cpu_packed_model(monkeypatch, request):
    # Enable the real packed host descriptors and range algorithm on CPU.
    # The server entry point itself always requires torch-npu.
    build = combo.open_engine
    monkeypatch.setattr(stage_engine, "packed_prefill_eligible", lambda *a, **kw: True)
    monkeypatch.setattr(stage_engine, "create_backend", lambda *a, **kw: cpu_attention())

    def host_engine(*args, **kwargs):
        engine = build(*args, **kwargs)
        engine.cache_manager._prepared_coverage = WeakKeyDictionary()
        return engine

    monkeypatch.setattr(combo, "open_engine", host_engine)
    torch.manual_seed(168)
    return (
        OuroForCausalLM(tiny_ouro_config(**getattr(request, "param", {}))).to(torch.bfloat16).eval()
    )


def test_three_arms_through_cpu_runner(cpu_packed_model):
    model = cpu_packed_model
    case = combo.Case(16, 2, 4, 2)
    donor, miss, targets = combo.prompts(case, model.config.vocab_size)
    for arm in combo.ARMS:
        result = combo.run_arm(model, case, arm, miss if arm == "miss" else donor, targets)
        assert result["hit_tokens"] == case.starts(arm)
        assert result["audit"]["packed_descriptors"] == 4
        assert result["audit"]["range_updates"] == 4 * 4 * 2
        assert result["prefill_rows"] == sum(18 - start for start in case.starts(arm).values())
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


def test_padded_linear_fp32_weights_bias_autocast_and_tail(monkeypatch):
    layer = nn.Linear(2, 3).to(torch.bfloat16).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2, 1], [0, -1], [1, 0]]))
        layer.bias.copy_(torch.tensor([1, 2, 3]))
    x = torch.arange(12, dtype=torch.bfloat16).reshape(2, 3, 2).transpose(0, 1)
    expected = layer(x)
    original_forward = layer.forward
    original_linear = linear.F.linear
    tiles = []

    def observe(tile, weight, bias):
        assert tile.dtype == weight.dtype == bias.dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        tiles.append(tile.clone())
        return original_linear(tile, weight, bias)

    with monkeypatch.context() as changes:
        changes.setattr(linear.F, "linear", observe)
        with linear.padded_fp32_linears(layer, 4) as (names, calls), torch.autocast("cpu"):
            actual = layer(x)
            assert layer(x[:0]).shape == (0, 2, 3)
            assert names == [""] and calls[""] == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [tuple(tile.shape) for tile in tiles] == [(4, 2), (4, 2)]
    torch.testing.assert_close(torch.cat(tiles)[:6], x.reshape(-1, 2).float())
    assert torch.count_nonzero(tiles[-1][2:]) == 0
    assert layer.forward == original_forward and layer.weight.dtype == torch.bfloat16
    # Restore modules installed before a later setup or execution failure.
    broken = nn.Sequential(layer, nn.Linear(3, 2))
    with pytest.raises(ValueError, match="BF16 checkpoint"), linear.padded_fp32_linears(broken, 4):
        pytest.fail("should reject mixed checkpoint weights")
    assert layer.forward == original_forward
    with pytest.raises(RuntimeError), linear.padded_fp32_linears(layer, 4):
        raise RuntimeError("interrupted")
    assert layer.forward == original_forward


def test_hf32_restored_after_linear_error():
    matmul = SimpleNamespace(allow_hf32=True)
    with pytest.raises(RuntimeError), linear._linear_math(matmul, "cpu"):
        assert matmul.allow_hf32 is False
        raise RuntimeError("interrupted")
    assert matmul.allow_hf32 is True


def test_fp32_only_combo_runs_all_linears_with_bf16_packed_sdpa(monkeypatch, cpu_packed_model):
    # Equal logits give deterministic IDs; the real engine still runs the projections.
    with torch.no_grad():
        cpu_packed_model.lm_head.weight.zero_()
    monkeypatch.setitem(combo.CASES, "short", combo.Case(16, 2, 4, 2))
    records = []

    def record(tag, *, echo=True, **data):
        records.append((tag, data))

    args = SimpleNamespace(linear_fp32_chunk=256, repeats=2)
    code = combo.run_model_cases(cpu_packed_model, args, ["short"], record, {}, False)
    assert code == 0
    arms = [data for tag, data in records if tag == "COMBO_ARM"]
    assert len(arms) == 6
    for arm in arms:
        assert arm["precision"] == "linear_fp32_padded" and arm["linear_fp32_chunk"] == 256
        assert arm["prefill_rows"] == (40 if arm["arm"] == "reuse" else 72)
        assert arm["linear_work"]["installed"] == 2 * 7 + 2
        assert set(arm["linear_work"]["uncalled"]) <= {"model.early_exit_gate"}
    assert records[-1][1]["failed"] == []
