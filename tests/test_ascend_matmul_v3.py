"""Controls for V3 device evidence and cross-kernel comparisons (CPU-only)."""

import csv
from copy import deepcopy

import pytest
import torch

from benchmarks import ascend_matmul_v3 as v3
from benchmarks.ascend_matmul_v3_compare import compare
from benchmarks.ascend_prefix_scenarios import summarize
from benchmarks.ascend_prefix_workload import Case, run_trial
from tests.helpers import tiny_ouro_model


def test_probe_checks_all_channels_and_orientation():
    x = torch.arange(576 * 24).reshape(576, 24).remainder(7).bfloat16()
    w = torch.eye(24).bfloat16()

    def shifted(x, weight):
        y = x @ weight.T
        if len(x) == 512:
            y[200, 0] += 1
        return y

    result = v3.collect(x, w, shifted)
    assert not result["invariant"]
    assert result["rows"][512]["different_elements_vs_576"] == 1
    assert result["rows"][512]["ch18_different_from_rounded_fp64"] == 0
    with pytest.raises(RuntimeError, match="one-hot"):
        v3.collect(x, w.roll(1, dims=0), lambda a, b: a @ b)


def test_verify_checks_device_type_even_if_name_claims_v3(tmp_path):
    path = tmp_path / "op_summary_01.csv"

    def write(last):
        with path.open("w", newline="") as stream:
            w = csv.writer(stream)
            w.writerow(["Op Name", "OP Type", "Input Shapes", "Output Shapes", "Output Data Types"])
            for m, kind in [(384, "MatMulV3"), (512, "MatMulV3"), (576, last)]:
                w.writerow(["MatMulV3", kind, f'"{m},2048;2048,2048"', f'"{m},2048"', "DT_BF16"])

    write("MatMulV2")
    with pytest.raises(RuntimeError, match="unverified"):
        v3.evidence(tmp_path, True)
    write("MatMulV3")
    assert len(v3.evidence(tmp_path, True)["types"]) == 3
    with pytest.raises(RuntimeError, match="unverified"):
        v3.evidence(tmp_path, True, [(1, 2048, 2048)])


def test_engine_intervention_and_failed_native_case(monkeypatch):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("tiny", "mix", 16, 3)

    def run(arm):
        return run_trial(model, "torch", case, arm, output_tokens=2, seed=168)

    native = {arm: run(arm) for arm in ("off", "miss", "reuse")}
    monkeypatch.setattr(v3, "load_projection", lambda _: torch.nn.functional.linear)
    layer = model.model.layers[0].self_attn
    original = layer.k_proj.forward
    with v3.use_v3_kv(model, None) as (calls, shapes):
        candidate = {arm: run(arm) for arm in native}
        assert calls["k_proj"] == calls["v_proj"] > 0
        assert shapes and layer.k_proj.forward != original
    assert layer.k_proj.forward == original
    native["reuse"]["requests"][0]["tokens"][0] += 1

    def report(trial):
        entry = dict(warmups=[], trials=[trial], summary=summarize(case, [trial]))
        return dict(state="complete", contract=dict(output_tokens=2), cases={case.name: entry})

    ref, new = report(native), report(candidate)
    new["contract"]["kv_projection"] = "MatMulV3"
    new["cases"][case.name]["v3_calls"] = calls
    row = compare(ref, new, [case.name])[case.name]
    assert row["native_parity"] == "FAIL" and row["v3_parity"] == "PASS"
    assert not any(row["v3_vs_native_off"].values())
    changed = deepcopy(new)
    changed["cases"][case.name]["trials"][0]["off"]["requests"][0]["tokens"][1] += 1
    row = compare(ref, changed, [case.name])[case.name]
    assert row["first_difference"]["off"]["token_1based"] == 2
