import csv
import json
from types import SimpleNamespace

from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks import ascend_pr1_failure_seeds as sweep
from benchmarks.ascend_main_prefix_results import Report, ids_digest
from tests.test_ascend_main_prefix_experiments import cpu_model  # noqa: F401


def test_seed_changes_targets_without_changing_planned_hits():
    for name in set(sweep.CASES["910B3"]) | set(sweep.CASES["910B4"]):
        case = matrix.FULL[name]
        historical = matrix.combo.workload(case, 1024)
        assert historical == matrix.combo.workload(case, 1024, seed=168)
        seen = {ids_digest(historical[2]["target-0"])}
        for seed in sweep.SEEDS:
            donor, miss, targets, _ = matrix.combo.workload(case, 1024, seed=seed)
            seen.add(ids_digest(targets["target-0"]))
            for rid, tokens in targets.items():
                assert tokens[0] != miss[0]
                start = case.starts("reuse")[rid]
                assert tokens[:start] == donor[:start]
                assert tokens[start] != donor[start]
        assert len(seen) == 11


def test_case_summary_preserves_failed_and_missing_seeds(tmp_path, cpu_model):  # noqa: F811
    case = "c4_h3"
    report = Report("CORRECTNESS", tmp_path / "seed-169")
    try:
        matrix.run_cases(
            SimpleNamespace(experiment="bf16", repeats=2, warmups=0),
            cpu_model,
            report,
            [case],
            modes=("main",),
            seed=169,
        )
    finally:
        report.stream.close()
    raw = [json.loads(line) for line in (report.folder / "raw.jsonl").read_text().splitlines()]
    event = next(row for row in raw if row["kind"] == "INPUTS")
    donor, miss, targets, _ = matrix.combo.workload(
        matrix.FULL[case], cpu_model.config.vocab_size, seed=169
    )
    assert event["seed"] == 169 and event["prompts"] == {
        rid: ids_digest(v) for rid, v in targets.items()
    }
    assert (event["donor_sha256"], event["miss_sha256"]) == (ids_digest(donor), ids_digest(miss))
    assert len([e for e in raw if e["kind"] == "TRIAL"]) == 6
    saved = next(e for e in raw if e["kind"] == "CASE_DETAILS")
    saved["all_issues"] = [
        dict(comparison="main.off_vs_main.reuse", round=2, request="target-2", first_diff_1based=7)
    ]
    (report.folder / "raw.jsonl").write_text("".join(json.dumps(e) + "\n" for e in raw))
    assert sweep.summarize(tmp_path, "910B3", "bf16", {0: [169, 170]}, workers_ok=False) == 2
    with (tmp_path / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert (rows[0]["status"], rows[0]["off_vs_reuse"], rows[0]["reuse_compared_requests"]) == (
        "FAIL",
        "1",
        "8",
    )
    assert rows[-1]["status"] == "INCOMPLETE" and not rows[-1]["off_vs_miss"]


def test_eight_cards_partition_ten_seeds_once(capsys):
    assert sweep.main(["--chip", "910B4", "--cards", "0,1,2,3,4,5,6,7", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert sorted(s for seeds in plan["assignments"].values() for s in seeds) == list(
        range(169, 179)
    )
    assert len(plan["cases"]) == 6 and plan["rounds"] == 2
