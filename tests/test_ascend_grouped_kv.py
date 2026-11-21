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
