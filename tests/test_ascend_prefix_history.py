from copy import deepcopy

import pytest

from benchmarks.ascend_prefix_history import compare_baseline
from benchmarks.ascend_prefix_scenarios import ARMS, summarize
from benchmarks.ascend_prefix_workload import Case


def measurement(case, repeats, ttft_ms):
    trials = []
    for _ in range(repeats):
        trial = {}
        for arm in ARMS:
            expected = case.prefix if arm == "reuse" else 0
            row = dict(
                prompt_sha256="abc",
                tokens=[7, 9],
                hit_tokens=expected,
                prefill_start=expected,
                expected_hit_tokens=expected,
                ttft_ms=ttft_ms,
                e2e_ms=2 * ttft_ms,
            )
            trial[arm] = dict(requests=[row], wall_ms=2 * ttft_ms, output_tokens=2)
        trials.append(trial)
    return {"summary": summarize(case, trials), "trials": trials}


def test_compare_historical_report_with_different_repeat_counts():
    case = Case("single", "length", 16, 2)
    contract = {"seed": 168, "checkpoint": "a"}
    prior = {
        "schema": 1,
        "contract": contract,
        "metadata": {"git_commit": "baseline", "checkpoint_path": "/other/machine"},
        "cases": {case.name: measurement(case, 1, 12)},
    }
    current = measurement(case, 3, 9)
    report = compare_baseline(contract, prior, **current)
    assert report["status"] == "PASS"
    for metrics in report["arms"].values():
        assert metrics["ttft_p50_saved_ms"] == 3
        assert metrics["ttft_p50_saved_pct"] == 25
        assert metrics["output_tok_s_gain_pct"] == pytest.approx(100 / 3)

    assert compare_baseline(contract, None, **current)["status"] == "NO_BASELINE"
    assert compare_baseline(contract | {"seed": 2}, prior, **current)["status"] == "INCOMPATIBLE"
    prior["cases"].clear()
    assert compare_baseline(contract, prior, **current)["status"] == "NO_BASELINE"


@pytest.mark.parametrize(
    ("defect", "status"),
    [
        ("prompt_sha256", "INCOMPATIBLE"),
        ("tokens", "FAIL"),
        ("truncated_log", "INCOMPATIBLE"),
        ("within_run_mismatch", "FAIL"),
    ],
)
def test_historical_comparison_withholds_deltas_on_bad_inputs(defect, status):
    case = Case("single", "length", 16, 2)
    current = measurement(case, 2, 10)
    prior = {"schema": 1, "contract": {}, "cases": {case.name: deepcopy(current)}}
    if defect == "truncated_log":
        prior["cases"][case.name]["trials"].pop()
    elif defect == "within_run_mismatch":
        current["summary"]["status"] = "FAIL"
    else:
        row = current["trials"][1]["reuse"]["requests"][0]
        row[defect] = "" if defect == "prompt_sha256" else [7, 2]
    report = compare_baseline({}, prior, **current)
    assert report["status"] == status
    assert "arms" not in report
    if defect == "tokens":
        assert report["first_diff_1based"] == 2
        assert (report["source"], report["repeat"], report["arm"]) == ("current", 2, "reuse")
