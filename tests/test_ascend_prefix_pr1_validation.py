"""Regression checks for attribution, diagnostic evidence and partial-run reporting."""

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks import ascend_main_prefix_service as service
from benchmarks import ascend_prefix_pr1_diagnostics as diagnostics
from benchmarks import ascend_prefix_pr1_suite as suite
from benchmarks.ascend_main_prefix_results import Report
from benchmarks.ascend_prefix_opportunities import opportunities
from tests.test_ascend_main_prefix_experiments import cpu_model  # noqa: F401


def test_main_only_matrix_keeps_three_arms_and_no_grouped_timings(tmp_path, cpu_model):  # noqa: F811
    report = Report("BENEFIT", tmp_path / "matrix")
    try:
        code = matrix.run_cases(
            SimpleNamespace(experiment="benefit", warmups=0, repeats=1),
            cpu_model,
            report,
            ["c4_h0"],
            modes=("main",),
        )
    finally:
        report.stream.close()
    assert code == 0
    saved = json.loads((report.folder / "review.json").read_text())
    assert saved["modes"] == ["main"]
    raw = [json.loads(line) for line in (report.folder / "raw.jsonl").read_text().splitlines()]
    trials = [e for e in raw if e["kind"] == "TRIAL"]
    assert {(e["mode"], e["arm"]) for e in trials} == {("main", arm) for arm in matrix.ARMS}
    paired = next(e["paired_ttft"] for e in raw if e["kind"] == "CASE_DETAILS")
    assert set(paired) == {"main_off_to_reuse", "main_miss_to_reuse", "main_off_to_miss"}


def test_drift_replay_binds_each_logit_row_to_returned_token(cpu_model):  # noqa: F811
    case = matrix.FULL["c4_h3"]
    inputs = matrix.combo.workload(case, cpu_model.config.vocab_size)
    result, records = diagnostics.trace_logits(cpu_model, case, inputs, "reuse")
    assert set(records) == set(result["requests"]) == set(inputs[2])
    for rid, rows in records.items():
        assert len(rows) == len(result["requests"][rid]["tokens"]) == 16
        assert all(len(hidden.shape) == len(scores.shape) == 1 for hidden, scores in rows)


def test_similar_opportunities_require_same_step_and_start():
    def step(*pairs):
        return dict(
            depths=[0, 1, 2, 3],
            members=[dict(prompt_sha256=k, start=s, query_rows=n) for k, s, n in pairs],
        )

    steps = [
        step(("a", 48, 23), ("b", 48, 20), ("c", 0, 20)),
        step(("d", 16, 16), ("e", 16, 16)),
        step(("f", 0, 10), ("g", 0, 10)),
        step(("h", 48, 20)),
    ]
    out = opportunities(steps, [dict(prompt_sha256=key) for key in "abcdefgh"])
    assert out["request_step_observations"] == 8
    assert out["already_regular_cold_members"] == 2
    assert out["additional_members_with_partner"] == {0: 2, 2: 2, 4: 4, 8: 4}
    with pytest.raises(AssertionError, match="missing"):
        opportunities([], [dict(prompt_sha256="target")])


def test_profile_requires_device_rows_and_full_artifacts(tmp_path):
    trace = [
        dict(name="aten::linear", ph="X", args={"Call stack": "model.py", "Input Dims": "8;32"})
    ]
    (tmp_path / "trace_view.json").write_text(json.dumps(trace))
    with pytest.raises(AssertionError, match="NPU kernels"):
        diagnostics.checked_profile_files(tmp_path, False)
    (tmp_path / "kernel_details.csv").write_text("Name,Duration(us)\nMatMul,2\n")
    assert diagnostics.checked_profile_files(tmp_path, False)["device_kernels"] == {"MatMul": 1}
    with pytest.raises(AssertionError, match="memory"):
        diagnostics.checked_profile_files(tmp_path, True)
    (tmp_path / "memory_record.csv").write_text("Size\n123\n")
    assert diagnostics.checked_profile_files(tmp_path, True)["stack_events"] == 1


def test_service_main_only_alternates_and_checks_historical_manifest(tmp_path, monkeypatch):
    class Bytes:
        def encode(self, text):
            return list(text.encode())

    monkeypatch.setattr(service, "load_tokenizer", lambda *_: Bytes())
    modes = []

    async def run_mode(args, report, mode, n, primers, plan, num_blocks, width):
        modes.append((n, mode))
        rows = [
            dict(
                id=r["id"],
                prompt_sha256=r["prompt_sha256"],
                ok=True,
                cohort=r["cohort"],
                tokens=[1] * r["output"],
                text_sha256="same",
                cached_tokens=0,
                prompt_tokens=r["prompt_tokens"],
                ttft_ms=10,
                output_tokens=r["output"],
            )
            for r in plan
        ]
        return rows, service.summary(rows, 1), []

    monkeypatch.setattr(service, "run_mode", run_mode)
    args = SimpleNamespace(
        main_only=True,
        observe_prefill=False,
        reference_traffic=None,
        model=tmp_path,
        requests=4,
        rps=0,
        seed=168,
        num_blocks=None,
        rounds=2,
    )
    report = Report("SERVICE", tmp_path / "first")
    try:
        assert asyncio.run(service.exercise(args, report)) == 0
    finally:
        report.stream.close()
    assert modes == [(1, "off"), (1, "main"), (2, "main"), (2, "off")]
    summary = json.loads((report.folder / "review.json").read_text())
    assert set(summary["paired_ttft"]) == {"off_to_main"}
    args.reference_traffic = report.folder / "traffic.json"
    args.seed += 1
    report = Report("SERVICE", tmp_path / "changed")
    try:
        with pytest.raises(AssertionError, match="workload differs"):
            asyncio.run(service.exercise(args, report))
    finally:
        report.stream.close()
    assert len(modes) == 4


def test_partial_or_skipped_runs_do_not_qualify_pr1(tmp_path):
    report = dict(full_pr1_scope=False, stages={"config": {"status": "PASS"}})
    assert suite.snapshot(tmp_path, report) == "PASS" and not report["full_pr1_pass"]
    report.update(full_pr1_scope=True)
    report["stages"]["profile"] = {"status": "NOT_RUN"}
    assert suite.snapshot(tmp_path, report) == "INCOMPLETE" and not report["full_pr1_pass"]
    xml = tmp_path / "skipped.xml"
    xml.write_text(
        '<testsuites><testsuite><testcase name="npu"><skipped/></testcase></testsuite></testsuites>'
    )
    stage = suite.Stage("empty-check", ["true"], xml)
    assert suite.run_stage(stage, tmp_path, {}, tmp_path)["status"] == "INCOMPLETE"


def test_profile_window_excludes_donor_and_measures_only_target_rows(cpu_model):  # noqa: F811
    case = matrix.FULL["p128_s8"]
    inputs = matrix.combo.workload(case, cpu_model.config.vocab_size)
    miss, reuse = (
        diagnostics.prefill_window(cpu_model, case, inputs, arm, nullcontext())
        for arm in ("miss", "reuse")
    )
    assert (miss["hit_tokens"], miss["prefill_rows"]) == (0, 136)
    assert (reuse["hit_tokens"], reuse["prefill_rows"]) == (128, 8)
    assert miss["prompt_sha256"] == reuse["prompt_sha256"]
