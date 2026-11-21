import io
import json
from copy import deepcopy

from benchmarks import ascend_grouped_kv_single as bench
from benchmarks.ascend_prefix_scenarios import ARMS, suites
from benchmarks.ascend_prefix_workload import expected_hits


def trial(case):
    arms = {}
    for arm in ARMS:
        primed, (hits,) = expected_hits(case, arm)
        row = dict(
            prompt_sha256="target",
            tokens=[7, 8],
            hit_tokens=hits,
            prefill_start=hits,
            expected_hit_tokens=hits,
        )
        arms[arm] = dict(
            requests=[row],
            prefix_hits=hits,
            prefix_queries=1,
            primed_hit_tokens=primed,
            expected_primed_hit_tokens=primed,
        )
    return {"native": arms, "grouped": deepcopy(arms)}


def test_full_selection_includes_all_single_request_controls():
    cases = bench.single_cases()
    assert cases == [case for case in suites()["full"] if case.concurrency == 1]
    assert len({case.name for case in cases}) == len(cases) == 23
    hits = {c.name: expected_hits(c, "reuse")[1][0] for c in cases}
    assert {name for name, count in hits.items() if not count} == {
        "edge_p0",
        "edge_p15",
        "total512_p0",
        "evicted_p64",
    }
    assert (hits["repeat_p128"], hits["total512_p512"], hits["resident_p64"]) == (112, 496, 64)


def test_single_suite_keeps_running_and_distinguishes_parity_from_cold_drift(monkeypatch):
    cases = bench.single_cases()[:3]

    def run(model, backend, case, grouped_matmul, output_tokens, seed):
        assert (output_tokens, seed) == (2, 42)
        trials = trial(case)
        if case == cases[0]:
            trials["grouped"]["reuse"]["requests"][0]["tokens"][1] = 10
        elif case == cases[1]:
            for arm in ARMS:
                trials["grouped"][arm]["requests"][0]["tokens"][0] = 11
        else:
            trials["native"]["reuse"]["requests"][0]["tokens"][1] = 12
        return trials, {arm: {1: 8} for arm in ARMS}

    monkeypatch.setattr(bench, "run_projection_trials", run)
    stream = io.StringIO()
    review = bench.run_suite(None, "torch", None, cases, 2, 2, 42, stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == 7 and rows[-1]["kind"] == "review"
    assert review["passed"] == 1 and review["failed"] == [c.name for c in cases[:2]]
    assert review["grouped_parity_failed"] == [cases[0].name]
    assert review["cold_baseline_changed"] == [cases[1].name]
    assert review["control_failed"] == []
    assert review["native_reuse_diverged"] == [cases[2].name]
    assert rows[0]["grouped_parity_diffs"][0]["first_diff_1based"] == 2
    assert not rows[0]["native_off_vs_grouped_off"]
    assert not rows[2]["grouped_parity_diffs"]
    assert rows[2]["native_off_vs_grouped_off"] == [[0, 1, 7, 11]]


def test_failed_hit_or_baseline_control_prevents_pass():
    case = bench.single_cases()[0]
    runs = trial(case)
    earlier = deepcopy(runs["native"]["off"])
    earlier["requests"][0]["tokens"][0] = 9
    runs["grouped"]["reuse"]["requests"][0]["prefill_start"] = 0
    checks = bench.check_trial(case, runs, earlier)
    assert not checks["passed"]
    assert {i["kind"] for i in checks["controls"]} == {
        "native_cold_changed_between_repeats",
        "hit_or_offset",
    }
