"""Regression tests for the joint K/V experiment and the public NPU call."""

import pytest
import torch

from benchmarks.ascend_grouped_kv import measure, operators, original_cases
from benchmarks.ascend_prefix_linear_repro import C1, C4


def operands(width):
    identity = torch.eye(width, dtype=torch.bfloat16)
    x = (torch.arange(24 * width).reshape(24, width) % 7).bfloat16()
    return x, {"k_proj": identity.roll(1, 1), "v_proj": -identity.roll(3, 1)}


def reference_grouped(xs, weights, *, split_item, group_type):
    assert xs[0] is xs[1]
    assert split_item == 0 and group_type == -1 and len(weights) == len(xs) == 2
    assert all(w.stride(0) == 1 for w in weights)
    return [x @ wt for x, wt in zip(xs, weights, strict=True)]


def test_known_values_and_two_original_geometries():
    x, weights = operands(32)
    ops = operators(weights, reference_grouped)
    report = measure(x, weights, ops, rows=(8, 16, 24), warmups=0, repeats=2)
    for proj in weights:
        assert all(report["summary"]["grouped"][proj].values())
        for case in report["cases"].values():
            assert case["details"]["grouped"][proj]["vs_native"] == 0
            assert len(case["pair_ms"]["native"]) == len(case["pair_ms"]["grouped"]) == 2
    cases = {
        C1: dict(x=x, info=dict(prefix_rows=8, controls=[])),
        C4: dict(
            x=x[:6].repeat(4, 1), info=dict(prefix_rows=4, slot_rows=6, focus_slot=2, controls=[])
        ),
    }
    original = original_cases(cases, weights, ops)
    assert original[C1]["grouped"]["k_proj"]["prefix_diffs"] == [0]
    assert original[C4]["grouped"]["v_proj"]["prefix_diffs"] == [0, 0, 0, 0]


def test_joint_call_checks_output_order_and_channels_beyond_18():
    x, weights = operands(32)

    def swapped(*args, **kwargs):
        return list(reversed(reference_grouped(*args, **kwargs)))

    with pytest.raises(ValueError, match="grouped/k_proj: one-hot"):
        measure(x, weights, operators(weights, swapped), rows=(8, 16, 24), repeats=1)

    def drift(*args, **kwargs):
        k, v = reference_grouped(*args, **kwargs)
        if len(k) == 16:
            k[:, 0] += 1  # Across shapes only.
            v[3, 1] += 1  # Within the same shape too.
        return k, v

    report = measure(x, weights, operators(weights, drift), rows=(8, 16, 24), warmups=0, repeats=1)
    result = report["cases"][16]["details"]["grouped"]
    assert (result["k_proj"]["within_m"], result["k_proj"]["vs_largest_m"]) == (0, 16)
    assert (result["v_proj"]["within_m"], result["v_proj"]["vs_largest_m"]) == (1, 1)
    assert not report["summary"]["grouped"]["v_proj"]["invariant"]
    assert report["summary"]["grouped"]["v_proj"]["ch18_matches_rounded"]


@pytest.mark.npu
@torch.inference_mode()
def test_npu_grouped_matmul_outputs_k_and_v():
    import torch_npu

    torch.npu.set_device(0)
    x, weights = operands(2048)
    x = x.repeat(16, 1).to("npu:0")
    ops = operators({k: v.to(x.device) for k, v in weights.items()}, torch_npu.npu_grouped_matmul)
    k, v = ops["grouped"](x)
    expected_k, expected_v = ops["native"](x)
    # Each row of each weight selects exactly one signed input, with no rounding ambiguity.
    assert torch.equal(k.cpu(), expected_k.cpu()) and torch.equal(v.cpu(), expected_v.cpu())


def test_grouped_engine_replay_consumes_both_outputs_and_restores_native():
    from benchmarks.ascend_grouped_kv_cases import replay
    from benchmarks.ascend_prefix_workload import Case
    from tests.helpers import tiny_ouro_model

    model = tiny_ouro_model(dtype=torch.bfloat16)
    baseline = model.model.layers[0].self_attn.k_proj.forward
    # Use short prompts for CPU coverage of donor prefill, cache hit, and decode.
    case = Case("c4_h4", "concurrency", 16, 4, concurrency=4, hits=4)
    result = replay(model, "torch", case, reference_grouped)
    assert result["summary"]["grouped_hits"] == [16] * 4
    assert result["summary"]["native_off_vs_grouped_off"] == []
    assert not result["summary"]["native_reproduced"]
    for arm, rows in result["grouped_calls_by_rows"].items():
        assert len(rows) >= 2
        original, joint = (result["trials"][impl][arm] for impl in ("native", "grouped"))
        assert [r["tokens"] for r in original["requests"]] == [
            r["tokens"] for r in joint["requests"]
        ]
    assert model.model.layers[0].self_attn.k_proj.forward == baseline


def test_grouped_split_observes_uncached_boundary_and_seventh_decision():
    from benchmarks.ascend_grouped_kv_cases import replay_split
    from benchmarks.ascend_prefix_workload import Case
    from tests.helpers import tiny_ouro_model

    case = Case("c4_h4", "concurrency", 16, 4, concurrency=4, hits=4)
    result = replay_split(tiny_ouro_model(dtype=torch.bfloat16), "torch", case, reference_grouped)
    info = result["summary"]
    assert info["controls"] == []
    assert not info["prior_grouped_focus_reproduced"] and not info["comparable_at_7"]
    assert info["hits"] == {"off": [0] * 4, "split": [0] * 4, "reuse": [16] * 4}
    assert info["focus_prefill"]["split"] == [(0, 16), (16, 4)]
    for mode, decision in info["focus_at_7"].items():
        assert decision["position"] == 7
        assert decision["selected_id"] == result["runs"][mode]["requests"][2]["tokens"][6]
