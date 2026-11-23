import json

import pytest

from benchmarks import ascend_prefix_pr1_replay as replay
from benchmarks import ascend_prefix_pr1_suite as suite
from benchmarks.ascend_main_prefix_linear import padded_fp32_linears
from benchmarks.ascend_main_prefix_results import Report
from tests.test_ascend_main_prefix_combo import cpu_packed_model  # noqa: F401


class TinyTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [2 + ord(letter) % 57 for letter in text]


@pytest.mark.parametrize("cpu_packed_model", [{"max_position_embeddings": 1200}], indirect=True)
def test_replay_checks_real_scheduler_and_attention_on_cpu(tmp_path, cpu_packed_model):  # noqa: F811
    # The fixture selects the Ascend eager implementation with CPU tensors.
    report = Report("REPLAY", tmp_path / "replay", linear_fp32_chunk=256)
    try:
        with padded_fp32_linears(cpu_packed_model, 256) as work:
            code = replay.replay(
                cpu_packed_model, TinyTokenizer(), report, replay.CASES, linear_work=work
            )
    finally:
        report.stream.close()
    result = json.loads((report.folder / "review.json").read_text())
    assert code == 0 and result["status"] == "PASS", result["failed"]
    assert result["precision"] == "linear_fp32_padded" and result["linear_fp32_chunk"] == 256
    cases = result["cases"]
    assert [(c["reference"], c["actual"]) for c in result["budget_comparisons"]] == [
        (f"short_full.{mode}.{rid}", f"short_split.{mode}.{rid}")
        for mode in ("off", "on")
        for rid in ("cold", "repeat", "branch")
    ]
    for name, repeat, branch in (
        ("short_full", 64, 48),
        ("short_split", 64, 48),
        ("long_full", 1088, 1056),
    ):
        case = cases[name]
        assert (
            case["controls_ok"] and [c["status"] for c in case["token_comparisons"]] == ["PASS"] * 5
        )
        for request, hits in (("cold", 0), ("repeat", repeat), ("branch", branch)):
            off, on = (case["runs"][mode][request] for mode in ("off", "on"))
            assert off["hit_tokens"] == 0 and on["hit_tokens"] == hits
            assert off["prompt_sha256"] == on["prompt_sha256"]
            assert on["live_blocks_after"] == 0 and not on["owns_allocation_after"]
    full = cases["short_full"]["runs"]["on"]["branch"]
    split = cases["short_split"]["runs"]["on"]["branch"]
    assert full["prompt_sha256"] == split["prompt_sha256"]
    assert full["prefill_spans"] == [[48, 71]]
    assert split["prefill_spans"] == [[48, 64], [64, 71]]
    assert cases["short_split"]["runs"]["off"]["branch"]["prefill_spans"] == [
        [0, 16],
        [16, 32],
        [32, 48],
        [48, 64],
        [64, 71],
    ]
    for step, shape in zip(split["attention_steps"], ("1x16x64", "1x7x71"), strict=True):
        assert list(step["sdpa_BxQxK"]) == [shape]
        assert step["explicit_masks_checked"] > 0
    long = cases["long_full"]["runs"]["on"]
    assert (
        long["cold"]["prompt_sha256"]
        == long["repeat"]["prompt_sha256"]
        != long["branch"]["prompt_sha256"]
    )


def test_failed_token_and_control_are_reported_separately(tmp_path, cpu_packed_model, monkeypatch):  # noqa: F811
    original = replay.run_request

    def corrupted(engine, name, prompt, expected_hit, budget, **kwargs):
        row = original(engine, name, prompt, expected_hit, budget, **kwargs)
        if name == "branch" and engine.cache_manager.enable_prefix_caching:
            row["token_ids"][2] = -1
        return row

    monkeypatch.setattr(replay, "run_request", corrupted)
    report = Report("REPLAY", tmp_path / "replay")
    try:
        assert replay.replay(cpu_packed_model, TinyTokenizer(), report, ["short_full"]) == 1
    finally:
        report.stream.close()
    result = json.loads((report.folder / "review.json").read_text())
    case = result["cases"]["short_full"]
    assert result["failed"] == ["short_full"] and case["controls_ok"]
    (diff,) = [
        item
        for item in case["token_comparisons"]
        if item["status"] == "FAIL" and item["actual"] == "on.branch"
    ]
    assert (diff["reference"], diff["first_diff_1based"], diff["actual_id"]) == (
        "off.branch",
        3,
        -1,
    )
    suite.markdown_report(
        tmp_path,
        dict(
            status="FAIL",
            full_pr1_pass=False,
            candidate="candidate",
            harness="harness",
            stages={"bf16-replay": {"status": "FAIL", "evidence": result}},
        ),
    )
    markdown = (tmp_path / "PR_RESULTS.md").read_text()
    assert "off.branch 与 on.branch 在第 3 个输出 token 首次不同" in markdown
    assert "| short_full | on | branch | 48 / 48 | [48,71) | 0 | PASS |" in markdown
    # Even a matching output must fail the path check if the expected hit is wrong.
    spec, budget = replay.CASES["short_full"]
    prompt = replay.prompt_pair(TinyTokenizer(), spec)["cold"]
    engine = replay.combo.open_engine(cpu_packed_model, spec, "off", token_budget=budget)
    try:
        row = original(engine, "cold", prompt, 16, budget, audit=False)
        assert {"hit_tokens", "prefill_spans"} <= set(row["controls_failed"])
    finally:
        engine.close()


def test_missing_or_truncated_output_cannot_pass_comparison():
    reference = dict(prompt_sha256="same", token_ids=[8, 9])
    assert replay.compare_tokens("off", reference, "on", {})["status"] == "NOT_RUN"
    short = reference | dict(token_ids=[8])
    mismatch = replay.compare_tokens("off", reference, "on", short)
    assert mismatch["status"] == "FAIL" and mismatch["first_diff_1based"] == 2
    assert mismatch["expected_id"] == 9 and mismatch["actual_id"] is None


@pytest.mark.parametrize("chip", ["910B3", "910B4"])
def test_both_replays_are_in_pr1_with_separate_precision(chip, capsys):
    assert suite.main(["all", "--list", "--chip", chip]) == 0
    lines = capsys.readouterr().out.splitlines()
    (bf16,) = [line for line in lines if line.startswith("bf16-replay ")]
    (fp32,) = [line for line in lines if line.startswith("fp32-replay ")]
    assert suite.REPLAY in bf16 and "--linear-fp32-chunk" not in bf16
    assert suite.REPLAY in fp32 and "--linear-fp32-chunk 256" in fp32
