"""Release plan, phase isolation and completeness of the measured evidence."""

import json
import threading
from collections import defaultdict

import pytest

from benchmarks import ascend_prefix_parallel as driver


@pytest.mark.parametrize("cards", [[3], [7, 2], list(range(8))])
def test_release_assignments_and_rounds(cards):
    jobs = driver.plan(("pr1", "pr2"), cards)
    assignments = defaultdict(list)
    for j in jobs:
        assert j.card in cards
        if j.precision == "stock":
            continue
        if j.cases or j.phase == "http":
            assert j.args[j.args.index("--correctness-rounds") + 1] == "2"
            repeats = "--rounds" if j.phase == "http" else "--repeats"
            assert j.args[j.args.index(repeats) + 1] == "3"
        for case in j.cases or (j.name,):
            assignments[j.project, case].append((j.precision, j.card))
    for rows in assignments.values():
        assert {p for p, _ in rows} == set(driver.PRECISIONS) and len(rows) == 3
        assert len({c for _, c in rows}) == 1
    assert all(("pr1", c) in assignments for c in driver.FULL)
    assert all(("pr2", c) in assignments for c in driver.GEOMETRIES)
    assert len({j.card for j in jobs if j.phase == "http"}) == min(2, len(cards))


def test_coverage_detects_repeat_and_missing_measurements(tmp_path):
    job = next(j for j in driver.plan(["pr2"], [0]) if j.phase == "http")
    rows = [
        dict(kind="SERVICE_RUN", round=n, mode=m)
        for n in (1, 2, 3)
        for m in ("main", "exact", "similar")
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    driver.check_rounds(job, tmp_path)
    rows[-1] = rows[0]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="round coverage"):
        driver.check_rounds(job, tmp_path)


def test_two_http_lanes_overlap_with_disjoint_cpus():
    jobs = [j for j in driver.plan(["pr1", "pr2"], [5, 2]) if j.phase == "http"]
    barrier = threading.Barrier(2, timeout=5)
    observed = {}

    def run(card, lane, cpus):
        barrier.wait()
        observed[card] = set(cpus)
        assert {j.card for j in lane} == {card}
        assert {j.project for j in lane} == {"pr1", "pr2"}

    driver.run_phase(jobs, run, [3, 5, 8, 9])
    assert not observed[5] & observed[2]
