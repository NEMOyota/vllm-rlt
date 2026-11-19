"""Verify the coordinates and controls of the independent BF16 projection replay."""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.ascend_prefix_linear_inputs import prepare_linear_inputs
from benchmarks.ascend_prefix_linear_ops import measure_linear
from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.ascend_prefix_workload import _prompts
from tests.helpers import tiny_ouro_model


def test_c1_detects_shape_difference_and_fixed_tiles_keep_real_rows(monkeypatch):
    # Make the first output channel depend solely on kernel M, with no noise.
    x = torch.ones((81, 3), dtype=torch.bfloat16)
    weight = torch.ones((2, 3), dtype=x.dtype)
    original = F.linear
    rows = []

    def tag_m(operand, w):
        rows.append(len(operand))
        result = original(operand, w)
        result[:, 0] = len(operand)
        return result

    monkeypatch.setattr(F, "linear", tag_m)
    options = dict(case="c1", prefix_rows=65, repeats=2, warmups=0)
    native = measure_linear(x, weight, **options)
    assert native["comparable"] and native["different_elements"] == [65]
    assert native["comparison"][0]["sample"] == dict(
        row=0, channel=0, reference=81.0, actual=65.0, fp64_dot=3.0
    )
    assert set(rows) == {65, 81}
    assert native["repeat_different_elements"] == {"full": [0], "prefix": [0]}
    assert all(len(t["samples"]) == 2 for t in native["timing_ms"].values())
    rows.clear()
    tiled = measure_linear(x, weight, op="tile64", **options)
    assert set(rows) == {64}
    assert tiled["comparable"] and tiled["different_elements"] == [0]
    assert tiled["work"]["prefix"] == dict(calls=2, active_rows=65, padded_rows=63)


def test_c4_measures_all_slots_against_slot_two_and_restricts_prefix(monkeypatch):
    slot_rows, prefix_rows = 6, 4
    x = torch.ones((4 * slot_rows, 3), dtype=torch.bfloat16)
    w = torch.ones((2, 3), dtype=x.dtype)
    original = F.linear

    def tag_slots(operand, weight):
        result = original(operand, weight)
        for slot, count in enumerate((1, 2, 0, 3)):
            start = slot * slot_rows
            result[start : start + count, 0] = 9
            if slot != 2:
                result[start + prefix_rows, 0] = 9
        return result

    monkeypatch.setattr(F, "linear", tag_slots)
    opts = dict(case="c4", slot_rows=slot_rows, prefix_rows=prefix_rows, repeats=2, warmups=0)
    result = measure_linear(x, w, **opts)
    assert result["comparable"] and result["input_slot_different_elements"] == [0, 0, 0, 0]
    assert result["different_elements"] == [1, 2, 0, 3]
    assert result["whole_slot_different_elements"] == [2, 3, 0, 4]
    x[0, 0] = 2
    mismatched = measure_linear(x, w, **opts)
    assert not mismatched["comparable"] and "slot_inputs_not_identical" in mismatched["controls"]


def test_same_shape_repeat_drift_invalidates_shape_comparison(monkeypatch):
    x = torch.ones((8, 2), dtype=torch.bfloat16)
    weight = torch.ones((3, 2), dtype=x.dtype)
    original = F.linear
    calls = []

    def drifting(operand, w):
        result = original(operand, w)
        calls.append(len(operand))
        result[0, 0] = len(calls)
        return result

    monkeypatch.setattr(F, "linear", drifting)
    run = measure_linear(x, weight, case="c1", prefix_rows=5, repeats=2, warmups=0)
    assert not run["comparable"]
    assert set(run["controls"]) == {"full:repeat_drift", "prefix:repeat_drift"}
    assert run["repeat_different_elements"] == {"full": [1], "prefix": [1]}


@pytest.mark.parametrize("name", ["total512_p384", "c4_h4"])
def test_fixture_uses_real_first_layer_inputs_and_weights(name):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = next(c for c in suites()["full"] if c.name == name)
    contract = {"seed": 168, "output_tokens": 16}
    x, weights, info = prepare_linear_inputs(model, case, contract)
    donor, targets, _ = _prompts(case, "reuse", 16, model.config.vocab_size, 168)
    tokens = targets[2] * 4 if name == "c4_h4" else donor
    layer = model.model.layers[0]
    with torch.inference_mode():
        expected = layer.input_layernorm(model.prelude(torch.tensor(tokens)))
    assert x.is_contiguous() and torch.equal(x.view(torch.int16), expected.view(torch.int16))
    assert weights["k_proj"] is layer.self_attn.k_proj.weight
    assert weights["v_proj"] is layer.self_attn.v_proj.weight
    assert info["prefix_rows"] == case.prefix and not info["controls"]
    assert info["input_control_different_elements"] == [0] * case.concurrency
