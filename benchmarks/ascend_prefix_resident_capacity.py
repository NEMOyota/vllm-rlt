"""Measure fixed-pool W4 residency with round-parallel Ascend devices and same-card paired arms."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import torch

from benchmarks.ascend_paged_prefill_probe import devices_arg
from benchmarks.ascend_prefix_benefit import Workload, workload_inputs
from benchmarks.ascend_resident_observer import Observer, dump_lines, residency
from benchmarks.ascend_resident_results import (
    GROUPS,
    Config,
    fixture,
    hot_count,
    report,
    stats,
    write_json,
)
from benchmarks.ascend_varlen_prefill import batch_run
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.device import resolve_device
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.models import OuroForCausalLM

MODULE = "benchmarks.ascend_prefix_resident_capacity"
ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", choices=GROUPS, nargs="+")
    p.add_argument("--case", choices=("hot", "cold", "mixed", "evict"), default="hot")
    p.add_argument("--order", choices=("interleaved", "hot-first", "cold-first", "all"))
    p.add_argument("--hot-requests", type=int, default=21, help="number of hot targets for mixed")
    p.add_argument("--mode", choices=("performance", "audit", "profile"), default="performance")
    p.add_argument("--seed", type=int, default=797543261)
    p.add_argument("--rounds", type=int, default=6)
    placement = p.add_mutually_exclusive_group()
    placement.add_argument("--device", type=int, help="run all rounds on one process-local NPU")
    placement.add_argument(
        "--cards",
        type=devices_arg,
        metavar="0,1,...",
        help="physical NPUs for round-robin placement of complete rounds",
    )
    p.add_argument(
        "--audit-cards",
        type=devices_arg,
        metavar="6,7",
        help="also run hot C/I and forced eviction I audits once, on two reserved cards",
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, help="new run directory")
    p.add_argument("--timeout-s", type=int, default=7200, help="wall limit per isolated arm")
    p.add_argument("--plan", action="store_true")
    for key, value in asdict(Config()).items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def plan(args):
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config())})
    cfg.validate()
    mask = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    visible = devices_arg(mask) if mask else None
    if args.device is not None:
        if args.device < 0 or (visible and args.device >= len(visible)):
            raise ValueError("--device is outside the process-visible devices")
        cards = [visible[args.device] if visible else args.device]
        worker_device = args.device
    else:
        cards = args.cards or [
            c for c in (visible or range(8)) if c not in (args.audit_cards or [])
        ]
        if not cards:
            raise ValueError("no cards available for performance rounds")
        if visible and not set(cards).issubset(visible):
            raise ValueError(f"--cards must be visible in ASCEND_RT_VISIBLE_DEVICES={mask}")
        worker_device = 0
    if args.rounds < 1 or args.timeout_s < 1:
        raise ValueError("rounds and timeout must be positive")
    groups = (
        args.groups
        or {"hot": list("ABCI"), "cold": list("CI"), "mixed": list("CBI"), "evict": ["I"]}[
            args.case
        ]
    )
    if len(set(groups)) != len(groups):
        raise ValueError("duplicate groups")
    n_hot = hot_count(cfg, args.case, args.hot_requests)
    order = args.order or {"mixed": "all", "evict": "cold-first"}.get(args.case, "interleaved")
    orders = ["interleaved", "cold-first", "hot-first"] if order == "all" else [order]
    if args.case == "evict" and (
        orders != ["cold-first"]
        or any(not GROUPS[g][0] or GROUPS[g][1] < cfg.num_blocks // cfg.full_blocks for g in groups)
    ):
        raise ValueError(
            "evict requires cache ON, cold-first and enough resident slots to fill the pool"
        )
    jobs = []
    for arrival in orders:
        for i in range(args.rounds):
            # Adjacent rounds reverse the order; rotate each subsequent pair.
            offset = (i // 2) % len(groups)
            sequence = groups[offset:] + groups[:offset]
            card = str(cards[i % len(cards)])
            for g in sequence[::-1] if i % 2 else sequence:
                path = f"{args.mode}/{args.case}-{arrival}/seed-{args.seed}/round-{i + 1:02d}/{g}"
                jobs.append(
                    dict(
                        case=args.case,
                        order=arrival,
                        group=g,
                        physical_card=card,
                        device=worker_device,
                        visible_devices=mask if args.device is not None else card,
                        round=i + 1,
                        seed=args.seed,
                        mode=args.mode,
                        path=path,
                        config=cfg.arm(g),
                        hot=n_hot,
                        theory=cfg.theory(g, n_hot),
                    )
                )
    if args.audit_cards is not None:
        if (
            args.mode != "performance"
            or len(args.audit_cards) != 2
            or set(cards).intersection(args.audit_cards)
        ):
            raise ValueError(
                "--audit-cards requires two dedicated cards outside the performance list"
            )
        for card, case, audit_groups in zip(
            args.audit_cards, ("hot", "evict"), (["C", "I"], ["I"]), strict=True
        ):
            audit_args = argparse.Namespace(
                **(
                    vars(args)
                    | dict(
                        case=case,
                        mode="audit",
                        groups=audit_groups,
                        order=None,
                        rounds=1,
                        audit_cards=None,
                        cards=[card],
                        device=None,
                    )
                )
            )
            jobs.extend(plan(audit_args)[1])
    return cfg, jobs


def fingerprint(checkpoint):
    index = checkpoint / "model.safetensors.index.json"
    names = ["config.json"]
    if index.exists():
        names.append(index.name)
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    if any(Path(n).name != n for n in shards):
        raise ValueError("invalid checkpoint shard name")
    files = {}
    for name in [*names, *shards]:
        digest = hashlib.sha256()
        with (checkpoint / name).open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                digest.update(chunk)
        files[name] = digest.hexdigest()
    return files


def memory(device, peak=False):
    fn = torch.npu
    if peak:
        return dict(
            peak_allocated_bytes=fn.max_memory_allocated(device),
            peak_reserved_bytes=fn.max_memory_reserved(device),
        )
    return dict(
        allocated_bytes=fn.memory_allocated(device), reserved_bytes=fn.memory_reserved(device)
    )


def warm(engine, model, cfg, seed, params, donor, enabled, evidence):
    prime = replace(params, max_tokens=1)
    w = Workload(
        cfg.input_length,
        cfg.output_length,
        cfg.warmup_requests,
        0,
        cfg.shared_prefix_length,
        cfg.warmup_requests,
    )
    warm_donor, targets, _ = workload_inputs(w, ~seed, model.config.vocab_size)
    evidence["warmup"] = {}
    for name, prompts, sampling in (
        ("warm_donor", [warm_donor], prime),
        ("warm_targets", targets, replace(params, max_tokens=min(2, cfg.output_length))),
    ):
        row = batch_run(engine, prompts, sampling, tag=name)
        evidence["warmup"][name] = row["wall_s"]
        if row["issues"]:
            raise RuntimeError(str(row["issues"]))
    # Reset only benchmark-owned cache references, after synchronization and a full drain.
    # Retain the warmed runtime and operator plans. No warmup KV enters target/primer.
    cache = engine.cache_manager
    cache.poll_prefixes()
    if cache._allocations or cache._pending_prefixes:
        raise RuntimeError("warmup did not drain")
    while cache._prefixes:
        _, blocks = cache._prefixes.popitem(last=False)
        cache._drop_refs(blocks)
    if len(cache._free_blocks) != cfg.num_blocks or any(cache._refs):
        raise RuntimeError("warmup left owned KV pages")
    evidence["primer_s"] = 0.0
    if enabled:
        row = batch_run(engine, [donor], prime, tag="donor")
        evidence["primer_s"] = row["wall_s"]
        if row["issues"] or row["hit_tokens"]:
            raise RuntimeError(f"invalid primer: {row['issues']} hits={row['hit_tokens']}")
        donor_keys = list(cache._prefix_keys(donor[: cfg.shared_prefix_length]))
        if not all(k in cache._prefixes for k in donor_keys):
            raise RuntimeError("primer did not publish the complete shared prefix")
        evidence["primer_physical_blocks"] = sum(len(cache._prefixes[k]) for k in donor_keys)


def summarize_target(obs, cfg, job, wall, records, used_before):
    errors = []
    requests = list(obs.requests.values())
    for rid, r in obs.requests.items():
        if (
            r.get("finish_reason") != "length"
            or len(r.get("tokens", [])) != cfg.output_length
            or r.get("exit_depths") != [cfg.max_loops] * cfg.output_length
        ):
            errors.append(f"{rid}: finish/length/depth mismatch")
    if errors:
        raise ValueError(str(errors))
    metrics = records["metrics"]
    for name, end, start in (
        ("queue", "admitted", "submitted"),
        ("execution_wait", "first_execution", "admitted"),
        ("ttft", "first_token", "submitted"),
        ("e2e", "finished", "submitted"),
        ("tpot", "finished", "first_token"),
    ):
        vals = []
        for r in requests:
            if name == "tpot" and len(r["tokens"]) < 2:
                continue
            ms = (r[end] - r[start]) * 1000 / (len(r["tokens"]) - 1 if name == "tpot" else 1)
            r[name + "_ms"] = ms
            vals.append(ms)
        metrics.update({f"{name}_{k}_ms": v for k, v in stats(vals).items()})
    prefill = [s for s in obs.steps if s["phase"] == "prefill"]
    decode = [s for s in obs.steps if s["phase"] == "recurrent"]
    rows = sum(s["q_rows"] for s in prefill)
    hits = sum(r["hits"] for r in requests)
    hit_requests = sum(r["hits"] > 0 for r in requests)
    spans = {}
    for step in prefill:
        for rid, start, count in step["spans"]:
            if start != spans.get(rid, obs.requests[rid]["hits"]):
                errors.append(f"noncontiguous actual prefill for {rid}")
            spans[rid] = start + count
    if (
        hits != obs.cache.prefix_hits - used_before
        or rows + hits != cfg.target_requests * cfg.input_length
        or any(spans.get(rid) != cfg.input_length for rid in obs.requests)
    ):
        errors.append("executed prefill spans and cache hit counters disagree")
    if any(r["hits"] and not r["intended_hot"] for r in requests):
        errors.append("cold input unexpectedly reused a prefix")
    expected = cfg.prefix_blocks // cfg.max_loops * cfg.block_size
    if (
        job["case"] == "hot"
        and job["config"]["cache_enabled"]
        and any(r["hits"] != expected for r in requests)
    ):
        records["workload_diagnostic"] = "full-hot W4 lost expected hits"
    if not job["config"]["cache_enabled"] and hits:
        errors.append("cache OFF recorded hits")
    if obs.cache.num_used_blocks or obs.cache._allocations:
        errors.append("live blocks/requests after drain")
    if metrics["resident_peak"] > job["theory"]["theory_peak"]:
        errors.append("observed residency exceeds physical/slot upper bound")
    if job["case"] == "evict":
        if (
            not metrics["cold_fill_observed"]
            or metrics["evicted_entries"] == 0
            or not any(r["intended_hot"] and r["hits"] == 0 for r in requests)
        ):
            errors.append("forced donor eviction followed by a cold recomputation was not observed")
    tokens = sum(len(r["tokens"]) for r in requests)
    inclusive = wall + records["primer_s"]
    paths = Counter()
    shapes = Counter()
    for s in obs.steps:
        paths.update(s["paths"])
        if s["phase"] in ("prefill", "recurrent"):
            shapes[
                s["phase"],
                s["batch_size"],
                s["q_rows"],
                s["max_context"],
                s.get("gather_kv_bytes_estimate_per_layer"),
            ] += 1
    records["execution_shapes"] = [
        dict(
            phase=phase,
            batch_size=size,
            q_rows=rows,
            max_context=context,
            gather_kv_bytes_estimate_per_layer=gather,
            steps=frequency,
        )
        for (phase, size, rows, context, gather), frequency in shapes.items()
    ]
    metrics.update(
        wall_s=wall,
        primer_s=records["primer_s"],
        inclusive_wall_s=inclusive,
        completed_requests=len(requests),
        output_tokens=tokens,
        output_tok_s=tokens / wall,
        request_s=len(requests) / wall,
        inclusive_output_tok_s=tokens / inclusive,
        inclusive_request_s=len(requests) / inclusive,
        hit_tokens=hits,
        prefill_rows=rows,
        input_hit_fraction=hits / (len(requests) * cfg.input_length),
        request_hit_fraction=hit_requests / len(requests),
        live_blocks_after=obs.cache.num_used_blocks,
        prefill_mean_batch=mean(s["batch_size"] for s in prefill),
        decode_mean_batch=mean(s["batch_size"] for s in decode) if decode else None,
        capacity_gap=job["theory"]["theory_peak"] - metrics["resident_peak"],
        **paths,
    )
    if errors:
        raise ValueError(str(errors))


def classify(exc):
    message = f"{type(exc).__name__} {exc}".lower()
    return (
        "OOM"
        if any(
            s in message for s in ("out of memory", "outofmemory", "acl_error_rt_memory_allocation")
        )
        else "ERROR"
    )


def target(engine, cfg, job, params, prompts, hot, work, records, device):
    cache = engine.cache_manager
    with ExitStack() as stack:
        stream = None
        if job["mode"] == "audit":
            stream = stack.enter_context((work / "kv_audit.jsonl").open("w", buffering=1))
        obs = Observer(engine, prompts, stream)
        for rid, row in obs.requests.items():
            row.update(
                request=rid,
                intended_hot=hot[rid],
                input_sha256=hashlib.sha256(json.dumps(prompts[rid]).encode()).hexdigest(),
            )
        obs.install(stack)
        if device is not None:
            torch.npu.reset_peak_memory_stats(device)
        used_before = cache.prefix_hits
        wall = None
        try:
            wall = obs.collect(prompts, params)
            records["outputs"] = {rid: r["tokens"] for rid, r in obs.requests.items()}
        finally:
            if wall is None:
                wall = max(1e-9, obs.now()) if obs.origin is not None else 1e-9
            curve, occupancy = residency(obs.events, wall, cfg.num_blocks)
            records["metrics"].update(**occupancy, **obs.counts)
            if device is not None:
                records["metrics"].update(memory(device, peak=True))
            # Preserve observations even if validation or the next step fails.
            dump_lines(work / "residency.jsonl", curve)
            dump_lines(work / "steps.jsonl", obs.steps)
            dump_lines(work / "requests.jsonl", obs.requests.values())
        if job["case"] == "evict":
            first_cold = cfg.num_blocks // cfg.full_blocks
            first_wave = [r for r in curve if r["event"] == "admit"][:first_cold]
            records["metrics"]["cold_fill_observed"] = (
                len(first_wave) == first_cold
                and first_wave[-1]["active"] == cfg.num_blocks
                and all(not hot[r["request"]] for r in first_wave)
            )
        summarize_target(obs, cfg, job, wall, records, used_before)
        dump_lines(work / "requests.jsonl", obs.requests.values())


@torch.inference_mode()
def worker(path):
    job = json.loads(path.read_text())
    work = path.parent
    cfg = Config(**{k: job["config"][k] for k in asdict(Config())})
    record = dict(status="RUNNING", stage="initialization", metrics={})
    write_json(work / "result.json", record)
    try:
        import torch_npu
        from torch_npu.utils.collect_env import get_cann_version

        device = resolve_device(f"npu:{job['device']}")
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32 = False
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        local_id = torch.npu.current_device()
        physical = visible.split(",")[local_id].strip() if visible else str(local_id)
        env = dict(
            card=torch.npu.get_device_name(job["device"]),
            physical_card=physical,
            process_device=f"npu:{local_id}",
            visible=visible,
            torch=str(torch.__version__),
            torch_npu=str(torch_npu.__version__),
            cann=str(get_cann_version()),
            allow_hf32=bool(torch.npu.matmul.allow_hf32),
        )
        write_json(work / "device.json", env)
        model = OuroForCausalLM.from_pretrained(
            job["checkpoint"], device=device, dtype=torch.bfloat16
        ).eval()
        record["model_config"] = model.config.to_dict()
        arm = job["config"]
        engine = StageEngine(
            model,
            attention_backend="ascend_eager",
            cache_config=CacheConfig(
                num_blocks=cfg.num_blocks,
                block_size=cfg.block_size,
                enable_prefix_caching=arm["cache_enabled"],
                layout="last_exited",
                incremental_allocation=False,
            ),
            execution_config=ExecutionConfig(
                multi_stream=False, use_uva=False, packed_prefill=True, ascend_paged_prefill=True
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=arm["resident_limit"],
                max_num_batched_tokens=cfg.max_num_batched_tokens,
                prefill_chunk_size=cfg.prefill_chunk_size,
            ),
        )
        try:
            cache = engine.cache_manager
            record["pool"] = dict(
                blocks=cache.num_blocks,
                shape=list(cache.key_cache.shape),
                bytes_per_block=cache.bytes_per_block,
                **memory(device),
            )
            if (
                cache.attention.packed_pool_prefill is None
                or cache.num_blocks != cfg.num_blocks
                or cache.storage_depths != cfg.max_loops
            ):
                raise RuntimeError("unexpected active backend or KV geometry")
            params = SamplingParams(
                temperature=0,
                min_loops=cfg.max_loops,
                max_loops=cfg.max_loops,
                max_tokens=cfg.output_length,
                ignore_eos=True,
                exit_threshold=1.0,
            )
            donor, prompts, hot = fixture(
                cfg, job["seed"], model.config.vocab_size, job["hot"], job["order"]
            )
            record["input_sha256"] = hashlib.sha256(
                json.dumps([donor, list(prompts.values())]).encode()
            ).hexdigest()
            record["request_ids"] = list(prompts)
            record["stage"] = "warmup_and_primer"
            write_json(work / "result.json", record)
            warm(engine, model, cfg, job["seed"], params, donor, arm["cache_enabled"], record)
            record["stage"] = "target"
            write_json(work / "result.json", record)
            target(engine, cfg, job, params, prompts, hot, work, record, device)
        finally:
            engine.close()
        if "workload_diagnostic" in record:
            record.update(status="DIAGNOSTIC", error=record["workload_diagnostic"])
        else:
            record["status"] = "OK"
    except Exception as exc:
        record.update(status=classify(exc), error=repr(exc), traceback=traceback.format_exc())
    write_json(work / "result.json", record)
    return 0 if record["status"] == "OK" else 1


async def run_child(work, spec, timeout):
    cmd = [sys.executable, "-m", MODULE, "--worker", str(spec)]
    job = json.loads(spec.read_text())
    if job["mode"] == "profile":
        cmd = [
            "msprof",
            f"--output={work / 'profiles'}",
            "--task-time=l1",
            "--ai-core=on",
            "--aic-mode=task-based",
            "--aic-metrics=PipeUtilization",
            *cmd,
        ]
    result = work / "result.json"
    try:
        with (work / "run.log").open("w") as log:
            env = os.environ.copy()
            if job.get("visible_devices") is not None:
                env["ASCEND_RT_VISIBLE_DEVICES"] = job["visible_devices"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                code = await asyncio.wait_for(process.wait(), timeout)
            except BaseException:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await process.wait()
                raise
        data = json.loads(result.read_text()) if result.exists() else {}
        if data.get("status") not in ("ERROR", "OOM", "DIAGNOSTIC") and (
            code or data.get("status") != "OK"
        ):
            raise RuntimeError(f"worker exited {code} without a successful result; see run.log")
        if (
            cmd[0] == "msprof"
            and data.get("status") == "OK"
            and not any((work / "profiles").rglob("PROF_*"))
        ):
            raise RuntimeError("msprof completed without a PROF_* capture")
    except asyncio.CancelledError:
        data = json.loads(result.read_text()) if result.exists() else {}
        write_json(result, data | {"status": "CANCELLED", "error": "launcher interrupted"})
        raise
    except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
        data = json.loads(result.read_text()) if result.exists() else {}
        data.update(
            status="TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else "ERROR",
            error=f"exceeded {timeout}s" if isinstance(exc, asyncio.TimeoutError) else repr(exc),
        )
        write_json(result, data)


async def run_cards(folder, jobs, timeout, env):
    # One serial queue per physical device. Only this parent writes the shared reports.
    lanes = defaultdict(list)
    for job in jobs:
        lanes[job["physical_card"]].append(job)
    env["devices"] = {}
    write_json(folder / "environment.json", env)

    async def run_lane(card, lane):
        for j in lane:
            work = folder / j["path"]
            work.mkdir(parents=True)
            spec = work / "job.json"
            write_json(spec, j)
            print(f"[card {card}] start {j['path']}", flush=True)
            await run_child(work, spec, timeout)
            if (work / "device.json").exists():
                device = json.loads((work / "device.json").read_text())
                env["devices"].setdefault(device["physical_card"], device)
                if len(lanes) == 1:
                    env["device"] = device
                write_json(folder / "environment.json", env)
            rows = report(folder, jobs)
            status = next(r["status"] for r in rows if r["path"] == j["path"])
            print(f"[card {card}] {status} {j['path']}", flush=True)

    loop = asyncio.get_running_loop()
    previous_term = signal.getsignal(signal.SIGTERM)
    loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    tasks = [asyncio.create_task(run_lane(card, lane)) for card, lane in lanes.items()]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous_term)
        rows = report(folder, jobs)
    return rows


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    try:
        cfg, jobs = plan(args)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        p.error(str(exc))
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return 0
    if not args.checkpoint or not args.checkpoint.is_dir() or not args.output:
        p.error("supply a checkpoint directory and a new output directory")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    for j in jobs:
        j["checkpoint"] = str(checkpoint)
    write_json(
        folder / "config.json",
        dict(
            parameters=vars(args) | {"checkpoint": str(checkpoint), "output": str(folder)},
            defaults=asdict(cfg),
            jobs=jobs,
        ),
    )
    env = dict(
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True
        ),
        checkpoint=str(checkpoint),
        files_sha256=fingerprint(checkpoint),
    )
    rows = asyncio.run(run_cards(folder, jobs, args.timeout_s, env))
    return int(any(r["status"] != "OK" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
