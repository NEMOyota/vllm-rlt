"""Exercise the production SDPA path with a tiny CPU Ouro and the service audit."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_service as service
from benchmarks import ascend_prefix_pr2 as pr2
from benchmarks.ascend_main_prefix_results import Report
from benchmarks.ascend_prefix_opportunities import opportunities
from tests.test_ascend_main_prefix_experiments import cpu_model  # noqa: F401


def test_tiny_ouro_all_six_workloads(tmp_path, monkeypatch, cpu_model):  # noqa: F811
    # The fixture uses the real scheduler and packed KV on CPU. Substitute the
    # NPU-only backend selection, then exercise the actual production kernel.
    build = combo.open_engine

    def host_engine(model, case, arm, *, prefix_prefill_slack=None):
        engine = build(model, case, arm)
        engine.cache_manager.attention.kernel.prefix_prefill_slack = prefix_prefill_slack
        return engine

    monkeypatch.setattr(combo, "open_engine", host_engine)
    report = Report("PR2_FIXTURE", tmp_path / "engine")
    try:
        result = pr2.evaluate(
            cpu_model,
            SimpleNamespace(linear_fp32_chunk=None, case=None, profile=False, repeats=1),
            report,
        )
    finally:
        report.stream.close()
    assert result == 0
    rows = [json.loads(line) for line in (report.folder / "raw.jsonl").read_text().splitlines()]
    near = [r for r in rows if r["kind"] == "TRIAL" and r["case"] == "near" and not r["warmup"]]
    assert len(near) == 3
    assert all(
        row["prefill_rows"] == 43 and set(row["hit_tokens"].values()) == {48} for row in near
    )
    similar = next(r for r in near if r["mode"] == "similar")
    assert similar["prefix_groups"]["saved_sdpa_calls"] == similar["layer_traversals"]


@pytest.mark.parametrize("padded", [0, 1])
def test_service_only_attributes_similar_savings_when_exercised(tmp_path, monkeypatch, padded):
    class Bytes:
        def encode(self, text):
            return list(text.encode())

    monkeypatch.setattr(service, "load_tokenizer", lambda *_: Bytes())
    orders = []

    async def fake_mode(args, report, mode, n, primers, plan, num_blocks, width):
        orders.append(mode)
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
                ttft_ms=2,
                output_tokens=r["output"],
            )
            for r in plan
        ]
        stats = service.summary(rows, 1)
        stats["prefix_groups"] = dict(
            groups=int(mode != "main"), padded_groups=padded if mode == "similar" else 0
        )
        return rows, stats, []

    monkeypatch.setattr(service, "run_mode", fake_mode)
    args = SimpleNamespace(
        main_only=False,
        pr2_slack=4,
        observe_prefill=False,
        reference_traffic=None,
        model=tmp_path,
        requests=4,
        rps=0,
        seed=168,
        num_blocks=None,
        rounds=3,
    )
    report = Report("SERVICE", tmp_path / "service")
    try:
        assert asyncio.run(service.exercise(args, report)) == 0
    finally:
        report.stream.close()
    review = json.loads((report.folder / "review.json").read_text())
    assert orders == [
        "main",
        "exact",
        "similar",
        "exact",
        "similar",
        "main",
        "similar",
        "main",
        "exact",
    ]
    assert ("exact_to_similar" in review["paired_ttft"]) == bool(padded)
    assert set(review["prefix_groups_per_round"]) == {"main", "exact", "similar"}


def test_opportunity_estimate_accounts_for_unequal_starts_and_padding():
    step = dict(
        depths=[0, 1, 2, 3],
        members=[
            dict(prompt_sha256="a", start=48, query_rows=80),
            dict(prompt_sha256="b", start=64, query_rows=72),
        ],
    )
    rows = opportunities([step], step["members"], row_bytes=(128, 64))
    prediction = rows["predicted_groups_per_layer_per_depth"]
    assert prediction[4]["members"] == 0 and prediction[8]["padded_groups"] == 1
