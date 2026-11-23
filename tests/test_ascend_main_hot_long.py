"""Exercise the replay, observed operators, and the clean/diagnostic split on CPU."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import ascend_main_hot_long as bench
from benchmarks.ascend_main_hot_long_audit import PrefillOps
from benchmarks.ascend_main_prefix_linear import padded_fp32_linears
from benchmarks.ascend_main_prefix_results import Report, ids_digest
from tests.helpers import tiny_ouro_config
from vllm_rlt.attention.backend import AttentionBackend, BackendCapabilities
from vllm_rlt.engine import stage_engine
from vllm_rlt.kernels.ascend_attention import AscendAttention
from vllm_rlt.models import OuroForCausalLM


def archive(tmp_path):
    base = [1 + i % 43 for i in range(242)]
    donor = base[:208] + [56] * 15

    def record(name, ids):
        return dict(
            id=name,
            prompt=" ".join(map(str, ids)),
            prompt_tokens=len(ids),
            prompt_sha256=ids_digest(ids),
        )

    target = dict(record("target-5", base), cohort="hot_long", min_primed=208)
    manifest = dict(kind="MANIFEST", primers=[record("primer_long", donor)], requests=[target])
    observed = dict(id="target-5", ok=True, prompt_sha256=ids_digest(base), tokens=[13, 24])
    runs = [
        dict(kind="SERVICE_RUN", mode=mode, requests=[dict(observed, cached_tokens=hit)])
        for mode, hit in (("off", 0), ("main", 208))
    ]
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [manifest, *runs]))
    tokenizer = SimpleNamespace(encode=lambda text: [int(t) for t in text.split()])
    return path, tokenizer, base


def test_restore_input_and_balanced_orders(tmp_path):
    path, tokenizer, base = archive(tmp_path)
    targets, donor, rid, first = bench.service_input(path, tokenizer, None)
    assert rid == "target-5" and first == 13 and targets[242] == base
    assert targets[257][:-1] == targets[256] and targets[256][:242] == base
    assert all(
        len(ids) == n and bench.shared_tokens(ids, donor) == 208 for n, ids in targets.items()
    )
    for n in range(3):
        orders = [bench.arm_order(i + 2 * n) for i in range(6)]
        assert len(set(orders)) == 6
        assert all(orders[i] == orders[i + 1][::-1] for i in (0, 2, 4))
    with pytest.raises(AssertionError, match="tokenizer"):
        bench.service_input(path, SimpleNamespace(encode=lambda text: [0] * len(text.split())), rid)


def test_event_pairs_are_read_after_one_device_fence(monkeypatch):
    state = dict(recorded=0, fences=0)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            state["recorded"] += 1
            self.index = state["recorded"]

        def elapsed_time(self, end):
            assert state["fences"] == 1 and end.index == self.index + 1
            return 0.25

    device = SimpleNamespace(type="npu")

    def synchronize(which):
        assert which is device
        state["fences"] += 1

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Event=Event, synchronize=synchronize), raising=False
    )
    ops = PrefillOps(device, 256, 242, 208)
    ops.forwards["q_proj", 34] = ops.tiles["q_proj", 34] = 1
    for name in ("k_read", "v_read", "sdpa"):
        assert ops._measure(name, abs, -1) == 1
    assert state == dict(recorded=6, fences=0)
    metrics = ops.result(1)
    assert metrics["kv_read_ms"] == 0.5 and metrics["sdpa_ms"] == 0.25


def test_one_request_all_lengths_and_target_only_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_engine, "packed_prefill_eligible", lambda *a, **kw: True)
    monkeypatch.setattr(
        stage_engine,
        "create_backend",
        lambda *a, **kw: AttentionBackend(
            AscendAttention(),
            BackendCapabilities(packed_prefill=True),
            dict(backend="ascend_eager", selected_backend="ascend_eager"),
        ),
    )
    torch.manual_seed(168)
    model = OuroForCausalLM(tiny_ouro_config(max_position_embeddings=300)).to(torch.bfloat16).eval()
    path, tokenizer, _ = archive(tmp_path)
    targets, donor, _, _ = bench.service_input(path, tokenizer, "target-5")
    report = Report("HOT_LONG", tmp_path / "result", linear_fp32_chunk=256)
    try:
        with padded_fp32_linears(model, 256) as work:
            token = bench.trial(model, targets[242], donor, "off", work)["first_token"]
            code = bench.run_cases(
                model,
                targets,
                donor,
                token,
                SimpleNamespace(repeats=1, audit_repeats=1),
                report,
                work,
            )
    finally:
        report.stream.close()
    result = json.loads((report.folder / "review.json").read_text())
    assert code == 0 and result["failed"] == []
    trials = [
        r
        for r in map(json.loads, (report.folder / "raw.jsonl").read_text().splitlines())
        if r["kind"] == "TARGET"
    ]
    assert len(trials) == 27 and len((report.folder / "cases.csv").read_text().splitlines()) == 10
    for r in trials:
        hit = 208 if r["arm"] == "reuse" else 0
        assert r["hit_tokens"] == hit and r["prefill_rows"] == r["length"] - hit
        if r["phase"] == "audit":
            ops = r["ops"]
            assert "ttft_ms" not in r and ops["clock"] == "cpu_wall_ms"
            assert ops["calls"] == dict(k_read=8, v_read=8, sdpa=8)
            tiles = 16 if r["length"] == 257 and not hit else 8
            assert ops["q_proj_tiles"] == tiles
            assert ops["kv_read_ms"] > 0 and ops["sdpa_ms"] > 0
            assert ops["kv_gather_bytes"]["k_read"] == ops["kv_gather_bytes"]["v_read"]
        else:
            assert "ops" not in r and r["ttft_ms"] > 0


def test_failed_diagnostics_preserve_clean_trials_and_suppress_benefit(tmp_path, monkeypatch):
    def fail(model, target, primer, arm, work, diagnostic):
        if diagnostic:
            raise AssertionError("missing KV gathers")
        return dict(first_token=42, ttft_ms=1.0)

    monkeypatch.setattr(bench, "trial", fail)
    path, tokenizer, _ = archive(tmp_path)
    targets, donor, _, _ = bench.service_input(path, tokenizer, None)
    report = Report("HOT_LONG", tmp_path / "result", linear_fp32_chunk=256)
    try:
        code = bench.run_cases(
            SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
            targets,
            donor,
            42,
            SimpleNamespace(repeats=1, audit_repeats=1),
            report,
            None,
        )
    finally:
        report.stream.close()
    res = json.loads((report.folder / "review.json").read_text())
    assert code == 1 and res["failed"] == [242, 256, 257]
    assert all(not c["paired_ttft"] for c in res["cases"].values())
    lines = list(map(json.loads, (report.folder / "raw.jsonl").read_text().splitlines()))
    assert sum(r.get("phase") == "timing" and "ttft_ms" in r for r in lines) == 9
