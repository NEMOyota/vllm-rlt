"""CPU regressions for mixed packed page tables, residency, fair pairing and SSE."""

import asyncio
from collections import Counter

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks import ascend_main_prefix_service as service
from benchmarks.ascend_main_prefix_grouping import GroupedAscendAttention
from tests.helpers import tiny_ouro_config
from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.engine import stage_engine
from vllm_rlt.kernels import ascend_attention
from vllm_rlt.models import OuroForCausalLM


def test_matrix_cases_and_paired_order():
    assert len(matrix.FULL) == 30 and list(matrix.FULL)[-1] == "c4_h3"
    for arm in matrix.ARMS:
        orders = Counter()
        for i in range(6):
            order = matrix.trial_order(i)
            assert len(set(order)) == 6
            positions = [n for n, (_, s) in enumerate(order) if s == arm]
            assert positions[1] == positions[0] + 1
            orders[order[positions[0]][0]] += 1
        assert orders == {"main": 3, "grouped": 3}


@pytest.fixture
def cpu_model(monkeypatch):
    monkeypatch.setattr(stage_engine, "packed_prefill_eligible", lambda *args, **kwargs: True)

    def make_backend(*args, **kwargs):
        return AttentionBackend(
            ascend_attention.AscendAttention(),
            BackendCapabilities(packed_prefill=True),
            dict(backend="ascend_eager", selected_backend="ascend_eager"),
        )

    monkeypatch.setattr(stage_engine, "create_backend", make_backend)
    torch.manual_seed(168)
    return OuroForCausalLM(tiny_ouro_config(max_position_embeddings=220)).to(torch.bfloat16).eval()


@pytest.mark.parametrize(
    "case", ["c4_h0", "c4_h1", "c4_h2", "c4_h3", "c4_h4", "resident_p64", "evicted_p64"]
)
def test_actual_engine_counters_and_residency(cpu_model, case):
    spec = matrix.FULL[case]
    inputs = matrix.combo.workload(spec, cpu_model.config.vocab_size)
    main = matrix.trial(cpu_model, spec, "main", "reuse", inputs)
    grouped = matrix.trial(cpu_model, spec, "grouped", "reuse", inputs)
    assert grouped["hit_tokens"] == main["hit_tokens"] == spec.starts("reuse")
    assert main["group_calls"] == {"cold": 0, "suffix": 0}
    assert not matrix.combo.token_diffs(main, grouped, "main_vs_grouped")
    if case.endswith("p64"):
        assert grouped["before_churn"] == 64
        assert grouped["hit_tokens"]["target-0"] == (64 if case.startswith("resident") else 0)


def test_packed_subgroups_keep_nonadjacent_query_rows_and_per_request_pages():
    # a,c are suffixes; b,d are cold. Nonadjacent groups and nonsequential private pages.
    base = ascend_attention.AscendAttention()
    grouped = GroupedAscendAttention()
    keys = [(12, 0)] * 2 + [(13, 0)] * 4 + [(14, 0)] * 2 + [(15, 0)] * 4
    lengths = [3, 4, 1, 2, 3, 4, 3, 4, 1, 2, 3, 4]
    g1 = base._plan(keys, lengths, "cpu", packed=True)
    g2 = grouped._plan(keys, lengths, "cpu", packed=True)
    q = torch.zeros(12, 1, 8)
    k = torch.zeros(8, 2, 1, 8)
    v = torch.arange(16, dtype=q.dtype)[:, None, None].expand(-1, 1, 8).contiguous().view_as(k)
    tables = torch.tensor([[7, 3], [2, 6], [5, 1], [0, 4]])
    flat = ascend_attention._flat_kv_rows(k, v)
    expected = base._forward(q, k, v, tables, g1, flat_rows=flat)
    actual = grouped._forward(q, k, v, tables, g2, flat_rows=flat)
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-6)
    assert (grouped.cold_prefill_calls, grouped.suffix_prefill_calls) == (1, 1)


def test_mix_and_sse_accounting():
    class Bytes:
        def encode(self, text):
            return list(text.encode())

    a = service.traffic(Bytes(), 32, 3, 168)
    assert a == service.traffic(Bytes(), 32, 3, 168)
    assert Counter(r["cohort"] for r in a[1]) == {
        k: 8 for k in ("hot_short", "cold_short", "hot_long", "cold_long")
    }
    assert a[1][0]["at_s"] == 0 and a[1][-1]["at_s"] > 0

    async def check():
        async def events(request):
            body = await request.json()
            assert body["stream_options"] == {"include_usage": True}
            data = (
                'data: {"choices":[{"text":"","finish_reason":null}]}\n\n'
                'data: {"choices":[{"text":"ok","finish_reason":"length"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            )
            if body["prompt"] == "success":
                data += "data: [DONE]\n\n"
            return web.Response(text=data, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/completions", events)
        async with TestClient(TestServer(app)) as cli:
            # The actual client uses absolute URLs; use its underlying session.
            url = str(cli.make_url("/")).rstrip("/")
            req = dict(id="x", prompt_sha256="p", cohort="hot", output=2, prompt_tokens=3)
            good = await service.completion(cli.session, url, "model", dict(req, prompt="success"))
            bad = await service.completion(cli.session, url, "model", dict(req, prompt="truncated"))
            assert good["ok"] and good["ttft_ms"] <= good["first_text_ms"] <= good["e2e_ms"]
            assert not bad["ok"] and "incomplete" in bad["error"]

    asyncio.run(check())


def test_service_checks_real_hits_and_generated_ids_independently_of_text():
    req = dict(id="target-0", prompt_sha256="abc", min_primed=16, max_possible=16, output=2)
    row = dict(id=req["id"], prompt_sha256="abc", ok=True, text_sha256="same")
    obs = dict(
        prompt_sha256="abc",
        tokens=[12, 9],
        exit_depths=[4, 4],
        cached_tokens=16,
        finish_reason="length",
    )
    state = dict(
        backend="ascend_eager",
        packed=True,
        dtype="torch.bfloat16",
        prefix_enabled=True,
        grouping_enabled=True,
        group_calls=dict(cold=0, suffix=0),
        prefix_queries=1,
        prefix_hits=16,
        live_blocks=0,
    )
    assert not service.reconcile("grouped", [], [req], [row], dict(state=state, requests=[obs]))
    assert row["cached_tokens"] == 16 and row["tokens"] == [12, 9]
    diffs = service.service_diffs([row], [dict(row, tokens=[12, 11])], "main_vs_grouped", 1)
    assert diffs[0]["first_diff_1based"] == 2 and diffs[0]["text_matches"]
    issues = service.reconcile(
        "grouped", [], [req], [dict(row)], dict(state=state, requests=[dict(obs, cached_tokens=0)])
    )
    assert len(issues) == 2


def test_inherited_bf16_drift_retains_trials_and_excludes_benefit(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from benchmarks.ascend_main_prefix_results import Report

    def inject(model, case, mode, arm, inputs):
        seq = [1] * 16 if arm != "reuse" else [2] * 16
        return dict(
            requests={
                "target-0": dict(
                    tokens=seq, ttft_ms=10, e2e_ms=100, hit_tokens=case.starts(arm)["target-0"]
                )
            },
            wall_ms=100,
        )

    monkeypatch.setattr(matrix, "trial", inject)
    report = Report("BENEFIT", tmp_path / "run")
    try:
        code = matrix.run_cases(
            SimpleNamespace(experiment="benefit", warmups=0, repeats=1),
            SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
            report,
            ["p128_s8"],
        )
    finally:
        report.stream.close()
    assert code == 1
    review = json.loads((report.folder / "review.json").read_text())
    assert set(review["failed_comparisons"]) == {
        "main.off_vs_main.reuse",
        "grouped.off_vs_grouped.reuse",
    }
    entries = [json.loads(line) for line in (report.folder / "raw.jsonl").read_text().splitlines()]
    assert len([r for r in entries if r["kind"] == "TRIAL"]) == 6
    assert next(r for r in entries if r["kind"] == "CASE_DETAILS")["paired_ttft"] == {}
