import csv
import json
from dataclasses import replace

import pytest
import torch

from benchmarks.ascend_prefix_resident_capacity import parser, plan, run_child, target
from benchmarks.ascend_resident_observer import audit_pages
from benchmarks.ascend_resident_results import GROUPS, Config, fixture, report, write_json
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM


def jobs(*args):
    return plan(parser().parse_args(args))[1]


def test_plan_w4_capacity_and_independent_budgets():
    all_jobs = jobs("--groups", *GROUPS, "--rounds", "1")
    assert [j["theory"]["theory_peak"] for j in all_jobs] == [4, 4, 5, 6, 8, 12, 15, 15, 15]
    assert [j["theory"]["theory_blocks_at_best_mix"] for j in all_jobs] == [
        4224,
        1920,
        5280,
        2496,
        3072,
        4224,
        5088,
        5088,
        5088,
    ]
    for j in all_jobs:
        cfg = j["config"]
        assert (
            cfg["num_blocks"],
            cfg["max_num_batched_tokens"],
            cfg["prefill_chunk_size"],
            cfg["warmup_requests"],
            cfg["target_requests"],
        ) == (5280, 1024, 256, 4, 32)
    custom = jobs(
        "--num-blocks", "6336", "--max-num-batched-tokens", "768", "--warmup-requests", "2"
    )
    assert all(
        j["config"]["num_blocks"] == 6336
        and j["config"]["max_num_batched_tokens"] == 768
        and j["config"]["prefill_chunk_size"] == 256
        and j["config"]["warmup_requests"] == 2
        for j in custom
    )
    normal = jobs()
    assert len(normal) == 24
    for i in range(0, 24, 8):
        assert [j["group"] for j in normal[i : i + 4]] == [
            j["group"] for j in reversed(normal[i + 4 : i + 8])
        ]


def test_orders_preserve_the_same_prompts_and_workload():
    cfg = Config()
    fixtures = [
        fixture(cfg, 797543261, 49152, 21, o) for o in ("interleaved", "cold-first", "hot-first")
    ]
    donor, original, hot = fixtures[0]
    assert sum(hot.values()) == 21
    for d, p, h in fixtures:
        assert d == donor and p == original and h == hot
        assert sum(len(v) for v in p.values()) == 131072
    assert list(fixtures[1][2].values()) == [False] * 11 + [True] * 21
    assert list(fixtures[2][2].values()) == [True] * 21 + [False] * 11
    assert len(jobs("--case", "mixed", "--rounds", "1")) == 9
    assert jobs("--case", "evict")[0]["hot"] == 27
    with pytest.raises(ValueError, match="cold-first"):
        jobs("--case", "evict", "--order", "hot-first")


def test_physical_counts_refcounts_and_illegal_alias():
    cache = KVCacheManager(1, 1, 4, 24, 2, 4, enable_prefix_caching=True)
    p = [1, 2, 3, 4, 5]
    assert cache.allocate("a", 5)
    cache.mark_imported_prefix("a", 5)
    cache.publish_prefix("a", p, 5)
    digests = list(cache._prefix_keys(p[:-1]))
    keys = {"a": digests, "b": digests}
    assert audit_pages(cache, keys) == dict(
        active=12, exclusive=12, shared=0, cache_only=0, raw_free=12
    )
    assert cache.allocate("b", 5, prefix=cache.lookup_prefix(p))
    assert audit_pages(cache, keys)["shared"] == 8
    with pytest.raises(ValueError, match="alias"):
        audit_pages(cache, {"a": digests, "b": [b"wrong", digests[1]]})
    cache.free("a")
    assert audit_pages(cache, keys)["exclusive"] == 12
    cache.free("b")
    assert audit_pages(cache, keys) == dict(
        active=0, exclusive=0, shared=0, cache_only=8, raw_free=16
    )


@pytest.mark.parametrize("case", ["hot", "evict"])
@torch.inference_mode()
def test_real_cpu_admission_release_and_observations(tmp_path, case):
    cfg = replace(
        Config(),
        num_blocks=24,
        input_length=32,
        output_length=2,
        shared_prefix_length=16,
        target_requests=4,
        prefill_chunk_size=16,
        max_num_batched_tokens=32,
    )
    j = jobs("--groups", "I", "--rounds", "1", "--mode", "audit")[0]
    hot = 4 if case == "hot" else 2
    j.update(case=case, config=cfg.arm("I"), theory=cfg.theory("I", hot))
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    eng = StageEngine(
        model,
        cache_config=CacheConfig(num_blocks=24, block_size=16, enable_prefix_caching=True),
        execution_config=ExecutionConfig(multi_stream=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=32, max_num_batched_tokens=32, prefill_chunk_size=16
        ),
    )
    params = SamplingParams(
        max_tokens=2, max_loops=4, min_loops=4, exit_threshold=1.0, ignore_eos=True, temperature=0
    )
    donor, prompts, intent = fixture(cfg, 123, 64, hot, "cold-first")
    try:
        eng.add_request("donor", donor, replace(params, max_tokens=1))
        while eng.has_unfinished_requests():
            eng.step()
        rec = {"metrics": {}, "primer_s": 1.2}
        target(eng, cfg, j, params, prompts, intent, tmp_path, rec, None)
        metrics = rec["metrics"]
        assert metrics["output_tokens"] == 8 and metrics["live_blocks_after"] == 0
        assert metrics["inclusive_output_tok_s"] < metrics["output_tok_s"]
        assert metrics["resident_peak"] == 2
        if case == "hot":
            assert metrics["hit_tokens"] == 64 and metrics["prefill_rows"] == 64
        else:
            assert metrics["evicted_entries"] > 0
        rows = [json.loads(s) for s in (tmp_path / "requests.jsonl").read_text().splitlines()]
        assert all(
            r["submitted"]
            <= r["admitted"]
            <= r["first_execution"]
            <= r["first_token"]
            <= r["finished"]
            for r in rows
        )
        assert all(r["exit_depths"] == [4, 4] for r in rows)
        curve = [json.loads(s) for s in (tmp_path / "residency.jsonl").read_text().splitlines()]
        assert all(r["active"] + r["cache_only"] + r["raw_free"] == 24 for r in curve)
        audit = [json.loads(s) for s in (tmp_path / "kv_audit.jsonl").read_text().splitlines()]
        events = {r["event"] for r in audit}
        assert {"admit", "release", "drained"} <= events
        if case == "evict":
            assert {"evict", "publish"} <= events
    finally:
        eng.close()


def make_results(tmp_path, planned):
    for j in planned:
        d = tmp_path / j["path"]
        d.mkdir(parents=True)
        write_json(
            d / "result.json",
            dict(
                status="OK",
                metrics={"output_tok_s": 10 if j["group"] == "A" else 12},
                request_ids=["r"],
                input_sha256="inputs",
                outputs={"r": [7, 8]},
            ),
        )


@pytest.mark.parametrize("mode", ["performance", "audit", "profile"])
def test_pairing_excludes_diagnostic_missing_and_nonperformance(tmp_path, mode):
    planned = jobs("--rounds", "3", "--groups", "A", "B", "--mode", mode)
    make_results(tmp_path, planned)
    fail = tmp_path / planned[-1]["path"] / "result.json"
    fail.unlink()
    report(tmp_path, planned)
    paired = list(csv.DictReader((tmp_path / "paired_summary.csv").open()))
    throughput = next(p for p in paired if p["metric"] == "output_tok_s")
    assert int(throughput["n"]) == (2 if mode == "performance" else 0)
    if mode == "performance":
        assert float(throughput["median_pct"]) == pytest.approx(20)
    changes = tmp_path / planned[1]["path"] / "result.json"
    data = json.loads(changes.read_text())
    data["outputs"]["r"][1] = 9
    write_json(changes, data)
    rows = report(tmp_path, planned)
    assert {"DIAGNOSTIC", "MISSING"} <= {r["status"] for r in rows}
    issues = [json.loads(s) for s in (tmp_path / "exceptions.jsonl").read_text().splitlines()]
    assert {r["comparison"] for r in issues if r["kind"] == "bf16_divergence"} == {
        "within_group_across_rounds",
        "cache_switch",
    }
    assert all(r["position_1based"] == 2 for r in issues if r["kind"] == "bf16_divergence")


def test_failed_worker_and_timeout_are_recorded(tmp_path, monkeypatch):
    from subprocess import TimeoutExpired

    job = tmp_path / "job.json"
    write_json(job, {"mode": "performance"})

    class Child:
        pid = 200

        def poll(self):
            return 0

        def wait(self, timeout=None):
            if timeout:
                raise TimeoutExpired("worker", timeout)
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: Child())
    run_child(tmp_path, job, 1)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "TIMEOUT"


@pytest.mark.parametrize("group", GROUPS)
def test_actual_w4_scheduler_first_wave_with_lightweight_cpu_pages(group):
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import Request

    cfg = Config()
    enabled, limit = GROUPS[group]
    donor, prompts, _ = fixture(cfg, 797543261, 49152, 32, "interleaved")
    # Exercise the real 4-plane index/allocator on CPU using the full W4 token corpus.
    # One small KV head keeps page addresses/counts exact without the model's 15.47-GiB tensors.
    cache = KVCacheManager(1, 1, 2, 5280, 16, 4, enable_prefix_caching=enabled)
    if enabled:
        assert cache.allocate("donor", 4096)
        cache.mark_imported_prefix("donor", 4096)
        cache.publish_prefix("donor", donor, 4096)
        cache.free("donor")
    sched = Scheduler(
        SchedulerConfig(max_num_seqs=limit, max_num_batched_tokens=1024, prefill_chunk_size=256),
        cache,
    )
    params = SamplingParams(max_tokens=128, min_loops=4, max_loops=4)
    for rid, tokens in prompts.items():
        sched.add_request(Request(rid, tokens, params))
    batch = sched.schedule()
    expect = cfg.theory(group, 32)
    assert len(cache._allocations) == expect["theory_peak"]
    assert cache.num_used_blocks == expect["theory_blocks_at_best_mix"]
    assert len(batch.requests) == 4 and batch.num_tokens == 1024
    assert {r.token_start for r in batch.requests} == ({3072} if enabled else {0})
    keys = {rid: list(cache._prefix_keys(prompts[rid][:-1])) for rid in cache._allocations}
    audit_pages(cache, keys)


@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_warmup_is_removed_and_only_on_runs_full_primer(monkeypatch, enabled):
    from benchmarks.ascend_prefix_resident_capacity import warm

    cfg = replace(
        Config(),
        num_blocks=24,
        input_length=32,
        output_length=2,
        shared_prefix_length=16,
        target_requests=4,
        warmup_requests=2,
    )
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    engine = StageEngine(
        model, cache_config=CacheConfig(num_blocks=24, block_size=16, enable_prefix_caching=enabled)
    )
    cache = engine.cache_manager
    # Historical batch_run expects this Ascend counter, irrelevant on the CPU fixture.
    monkeypatch.setattr(cache.attention.kernel, "sdpa_calls", 0, raising=False)
    donor, _, _ = fixture(cfg, 123, 64, 4, "interleaved")
    params = SamplingParams(
        max_tokens=2, max_loops=4, min_loops=4, exit_threshold=1.0, ignore_eos=True, temperature=0
    )
    record = {}
    try:
        warm(engine, model, cfg, 123, params, donor, enabled, record)
        assert set(record["warmup"]) == {"warm_donor", "warm_targets"}
        assert record["primer_s"] > 0 if enabled else record["primer_s"] == 0
        assert set(cache._prefixes) == (set(cache._prefix_keys(donor[:-1])) if enabled else set())
        assert not cache._allocations
    finally:
        engine.close()


def test_operator_audit_counts_dispatched_shapes_without_replacing_results():
    from collections import Counter
    from contextlib import ExitStack
    from types import SimpleNamespace

    from benchmarks.ascend_resident_observer import audit_attention
    from vllm_rlt.kernels import ascend_attention

    q = torch.randn(3, 2, 1, 4)
    k = torch.randn(3, 2, 7, 4)
    expected = ascend_attention._attend(q, k, k)
    strategy = SimpleNamespace(fused_attention=lambda q, k, v, **kw: q)
    counts = Counter()
    with ExitStack() as stack:
        audit_attention(stack, strategy, counts)
        torch.testing.assert_close(ascend_attention._attend(q, k, k), expected)
        assert (
            strategy.fused_attention(
                q, k, k, actual_seq_lengths=[2, 7], actual_seq_lengths_kv=[9, 5]
            )
            is q
        )
    assert counts == {("sdpa", (1, 1, 1), (7, 7, 7)): 1, ("fia", (2, 5), (9, 5)): 1}
