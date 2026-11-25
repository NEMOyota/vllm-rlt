import csv
import json
from types import SimpleNamespace

import pytest

from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks import ascend_pr1_failure_seeds as sweep
from benchmarks import ascend_pr1_seed_evidence as evidence
from benchmarks.ascend_main_prefix_results import Report, ids_digest
from tests.test_ascend_main_prefix_experiments import cpu_model  # noqa: F401


def test_seed_changes_targets_without_changing_planned_hits():
    for name in set(sweep.CASES["910B3"]) | set(sweep.CASES["910B4"]):
        case = matrix.FULL[name]
        historical = matrix.combo.workload(case, 1024)
        assert historical == matrix.combo.workload(case, 1024, seed=168)
        seen = {ids_digest(historical[2]["target-0"])}
        for seed in (*sweep.SEEDS, *sweep.BF16_SEEDS_30):
            donor, miss, targets, _ = matrix.combo.workload(case, 1024, seed=seed)
            seen.add(ids_digest(targets["target-0"]))
            for rid, tokens in targets.items():
                assert tokens[0] != miss[0]
                start = case.starts("reuse")[rid]
                assert tokens[:start] == donor[:start]
                assert tokens[start] != donor[start]
        assert len(seen) == 41


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
    report = evidence.collect(tmp_path, case)
    assert report["summary"]["incomplete"] == 9
    assert [row["seed"] for row in report["rows"]] == ["169", "170"]
    details = report["failed_seed_evidence"][169]
    assert len([event for event in details if event["kind"] == "TRIAL"]) == 6
    assert (
        next(event["all_issues"] for event in details if event["kind"] == "CASE_DETAILS")
        == saved["all_issues"]
    )


def test_eight_cards_partition_ten_seeds_once(capsys):
    assert sweep.main(["--chip", "910B4", "--cards", "0,1,2,3,4,5,6,7", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert sorted(s for seeds in plan["assignments"].values() for s in seeds) == list(
        range(169, 179)
    )
    assert len(plan["cases"]) == 6 and plan["rounds"] == 2


@pytest.mark.parametrize(
    ("chip", "cards", "expected_cases"),
    [
        ("910B3", "0,1,2,3,4,5,6,7", ["c4_h3", "c4_h4", "p128_s128", "p384_s8", "p512_s8"]),
        ("910B4", "0", ["c4_h3", "c4_h4", "p128_s128", "p1024_s8", "p256_s8", "total512_p384"]),
    ],
)
def test_stratified_bf16_replay_plan(capsys, chip, cards, expected_cases):
    assert sweep.main(["--chip", chip, "--cards", cards, "--bf16-30-seeds", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    seeds = plan["seeds"]
    assert len(seeds) == len(set(seeds)) == 30 and not set(seeds).intersection(range(168, 179))
    span = 2**31
    for i, seed in enumerate(sorted(seeds)):
        assert i * span // 30 <= seed < (i + 1) * span // 30
    assert seeds == list(sweep.BF16_SEEDS_30)
    assert sorted(s for batch in plan["assignments"].values() for s in batch) == sorted(seeds)
    assert plan["cases"] == expected_cases
    assert plan["precision"] == "bf16" and plan["cann"] == "9.0.0"
    assert plan["arms"] == ["off", "miss", "reuse"] and plan["rounds"] == 2


@pytest.mark.parametrize(
    "options",
    [
        ["--precision", "fp16"],
        ["--precision", "fp32-linear"],
        ["--case", "p1024_s8"],
        ["--seed", "168"],
    ],
)
def test_fixed_bf16_campaign_rejects_overrides(options):
    with pytest.raises(SystemExit) as error:
        sweep.main(["--chip", "910B3", "--bf16-30-seeds", "--plan", *options])
    assert error.value.code == 2


def test_b4_failure_can_be_replayed_on_b3_on_one_selected_card(capsys):
    assert (
        sweep.main(
            ["--chip", "910B3", "--cards", "5,3", "--case", "p1024_s8", "--seed", "168", "--plan"]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == ["p1024_s8"] and plan["seeds"] == [168]
    assert plan["assignments"] == {"5": [168]}
