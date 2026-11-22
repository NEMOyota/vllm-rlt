from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks import ascend_prefix_framework_regression as bench


def test_reference_policy_preserves_the_29_archived_cases():
    cases = bench.suites()["full"]
    assert len(cases) == 29 and bench.C4_H3.name not in {c.name for c in cases}
    assert (
        bench.C4_H3.concurrency,
        bench.C4_H3.hits,
        bench.C4_H3.reusable_tokens,
        bench.C4_H3.suffix,
    ) == (4, 3, 128, 16)
    c = next(c for c in cases if c.name == "c4_h4")
    archive = {
        "state": "complete",
        "contract": {"seed": 168, "output_tokens": 16},
        "cases": {
            c.name: {
                "trials": [{"off": {"requests": []}}],
                "summary": {"case": asdict(c), "status": "PASS"},
            }
        },
    }
    refs = bench.references(archive, [bench.C4_H3, c], 256)
    assert refs == {"c4_h3": None, "c4_h4": {"requests": []}}
    archive["cases"].clear()
    with pytest.raises(ValueError, match="c4_h4 is missing"):
        bench.references(archive, [c], 256)


def test_failed_or_incomplete_case_has_no_reported_latency():
    trial = SimpleNamespace(
        case_name="c4_h3",
        gates=1,
        repeats=1,
        counts=dict.fromkeys(bench.paired.ARMS, 1),
        failures=[],
    )
    assert bench.case_review(trial)["status"] == "INCOMPLETE"
    trial.gates = 2
    trial.counts = dict.fromkeys(bench.paired.ARMS, 2)
    trial.failures = [{"repeat": 1, "issues": [{"check": "suffix_calls"}]}]
    report = bench.case_review(trial)
    assert (
        report["status"] == "FAIL" and report["failures"] == trial.failures and "ttft" not in report
    )


def test_slowdowns_are_paired_per_arm_and_only_for_passing_cases():
    measurements = {
        arm: {"paired_saved_ms": values, "median_paired_saved_ms": ms}
        for arm, values, ms in (
            ("off", [-2, -1, -4], -2),
            ("miss", [1, -2, -3], -2),
            ("reuse", [-7, -6, -5], -6),
        )
    }
    slow = bench.slower_in_every_repeat(
        {"c4_h3": {"status": "PASS", "ttft": measurements}, "failed": {"status": "FAIL"}}
    )
    assert [(s["arm"], s["extra_ms"]) for s in slow] == [("reuse", 6), ("off", 2)]
