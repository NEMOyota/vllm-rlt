"""Replay the same mixed, streamed completions through fresh off/main/grouped services."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import aiohttp

from benchmarks.ascend_main_prefix_combo import BLOCK, SEED
from benchmarks.ascend_main_prefix_linear import (
    add_precision_arg,
    benchmark_dtype,
    check_linear_usage,
)
from benchmarks.ascend_main_prefix_results import Report, ids_digest, quantiles, savings, write_csv
from benchmarks.ascend_prefix_combo_checks import require
from benchmarks.ascend_prefix_opportunities import opportunities
from vllm_rlt.models import load_tokenizer

MODES = ("off", "main", "grouped")
TIMEOUT = 300


def shared_tokens(a, b):
    limit = min(len(a), len(b)) - 1
    different = next((i for i, (x, y) in enumerate(zip(a[:limit], b[:limit])) if x != y), limit)
    return max(0, different) // BLOCK * BLOCK


def traffic(tokenizer, count, rps, seed):
    rng = random.Random(seed)

    # Keep families intentionally distinct before their first page. The private
    # suffix is unique even when the tokenizer merges a word at the boundary.
    def context(size):
        return "Review the warranty and list the applicable conditions. " * (
            5 if size == "short" else 24
        )

    primers = [
        dict(
            id="warmup",
            cohort="primer",
            prompt="Warm the server with a brief unrelated status check.",
            output=4,
        )
    ]
    for size in ("short", "long"):
        primers.append(
            dict(
                id=f"primer_{size}",
                cohort="primer",
                prompt=f"{size} manual. " + context(size) + "Primer only.",
                output=1,
            )
        )
    plan = []
    arrival = 0.0
    for i in range(count):
        if i % 4 == 0:
            a, b = ("short", "long") if (i // 4) % 2 == 0 else ("long", "short")
            block = [("hot", a), ("cold", b), ("hot", a), ("cold", b)]
            rng.shuffle(block)
        heat, size = block[i % 4]
        nonce = f"{rng.getrandbits(64):016x}"
        pre = f"{size} manual. " if heat == "hot" else f"{nonce} private {size} note. "
        prompt = pre + context(size) + f"Question {i}: summarize item {nonce}."
        if i and rps:
            arrival += rng.expovariate(rps)
        plan.append(
            dict(
                id=f"target-{i}",
                cohort=f"{heat}_{size}",
                prompt=prompt,
                output=8 if size == "short" else 16,
                at_s=arrival,
            )
        )
    seen = set()
    token_lists = {}
    for req in primers + plan:
        tokens = list(tokenizer.encode(req["prompt"]))
        req.update(prompt_tokens=len(tokens), prompt_sha256=ids_digest(tokens))
        require(req["prompt_sha256"] not in seen, "two request texts tokenize identically")
        seen.add(req["prompt_sha256"])
        token_lists[req["id"]] = tokens
    for req in plan:
        ids = token_lists[req["id"]]
        # The first paid reuse comes from a primer; concurrent arrivals may
        # expose extra complete pages. Predict a range without assuming order.
        lower = max(shared_tokens(ids, token_lists[p["id"]]) for p in primers)
        upper = max(
            [lower]
            + [
                shared_tokens(ids, tokens)
                for rid, tokens in token_lists.items()
                if rid != req["id"]
            ]
        )
        require(
            lower > 0 if req["cohort"].startswith("hot") else upper == 0,
            f"tokenizer did not construct intended {req['cohort']} for {req['id']}",
        )
        req.update(min_primed=lower, max_possible=upper)
    return primers, plan


async def completion(client, url, model, req):
    payload = dict(
        model=model,
        prompt=req["prompt"],
        max_tokens=req["output"],
        temperature=0,
        min_loops=4,
        max_loops=4,
        exit_threshold=1.0,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    clock = time.perf_counter
    sent = clock()
    row = dict(id=req["id"], cohort=req["cohort"], prompt_sha256=req["prompt_sha256"], sent=sent)
    try:
        async with client.post(url + "/v1/completions", json=payload) as response:
            row["http"] = response.status
            if response.status != 200:
                raise ValueError((await response.text())[:300])
            require(response.content_type == "text/event-stream", "expected SSE")
            first = first_text = None
            finish = usage = None
            done = False
            chunks = []
            async for raw in response.content:
                line = raw.decode().strip()
                if not line or line.startswith(":"):
                    continue
                require(line.startswith("data:") and not done, "invalid SSE sequence")
                item = line[5:].strip()
                if item == "[DONE]":
                    done = True
                    continue
                event = json.loads(item)
                require("error" not in event, "in-band error")
                choices = event["choices"]
                if choices:
                    require(
                        finish is None and usage is None and len(choices) == 1,
                        "unexpected token event",
                    )
                    now = clock()
                    if first is None:
                        first = now
                    value = choices[0]["text"]
                    if value and first_text is None:
                        first_text = now
                    chunks.append(value)
                    finish = choices[0]["finish_reason"]
                if event.get("usage") is not None:
                    require(
                        not choices and finish is not None and usage is None,
                        "unexpected usage event",
                    )
                    usage = event["usage"]
            ended = clock()
            require(
                done and first is not None and finish == "length" and usage is not None,
                "incomplete SSE completion",
            )
            require(
                usage["completion_tokens"] == req["output"]
                and usage["prompt_tokens"] == req["prompt_tokens"],
                f"unexpected HTTP usage {usage}",
            )
            row.update(
                ok=True,
                ttft_ms=1000 * (first - sent),
                first_text_ms=1000 * (first_text - sent) if first_text else None,
                e2e_ms=1000 * (ended - sent),
                end=ended,
                output_tokens=usage["completion_tokens"],
                prompt_tokens=usage["prompt_tokens"],
                text_sha256=hashlib.sha256("".join(chunks).encode()).hexdigest(),
            )
    except (
        aiohttp.ClientError,
        TimeoutError,
        ValueError,
        KeyError,
        AssertionError,
        TypeError,
    ) as exc:
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}", end=clock())
    return row


@asynccontextmanager
async def service(command, log, env, client, url, model, timeout):
    with log.open("x") as out:
        proc = subprocess.Popen(
            command, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; see {log}")
                try:
                    async with client.get(
                        url + "/health", timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            async with client.get(
                                url + "/v1/models", timeout=aiohttp.ClientTimeout(total=2)
                            ) as models:
                                require(
                                    [m["id"] for m in (await models.json())["data"]] == [model],
                                    "port belongs to another server",
                                )
                            break
                except (aiohttp.ClientConnectionError, TimeoutError):
                    pass
                await asyncio.sleep(0.25)
            else:
                raise TimeoutError(f"server readiness timeout; see {log}")
            yield
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                await asyncio.to_thread(proc.wait)
        if proc.returncode:
            raise RuntimeError(f"server exited {proc.returncode} at shutdown; see {log}")


async def burst(client, url, model, plan, concurrency):
    slots = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async def send(req):
        due = start + req["at_s"]
        await asyncio.sleep(max(0, due - time.perf_counter()))
        async with slots:
            row = await completion(client, url, model, req)
        row["queue_ms"] = 1000 * (row["sent"] - due)
        if row["ok"]:
            row["arrival_ttft_ms"] = row["queue_ms"] + row["ttft_ms"]
            row["arrival_e2e_ms"] = 1000 * (row["end"] - due)
        return row

    rows = await asyncio.gather(*(send(req) for req in plan))
    wall = max(r["end"] for r in rows) - start
    for row in rows:
        row.pop("sent")
        row.pop("end")
    return rows, wall


def reconcile(mode, primers, plan, rows, audit, *, linear_fp32_chunk=None):
    state = audit["state"]
    require(
        state.get("backend") == "ascend_eager"
        and state["packed"]
        and state["dtype"] == state["value_cache_dtype"] == str(benchmark_dtype())
        and state["linear_weight_dtypes"] == [str(benchmark_dtype())]
        and not state["async_scheduling"]
        and not state["multi_stream"]
        and state["prefix_enabled"] == (mode != "off")
        and state["grouping_enabled"] == (mode == "grouped")
        and state.get("linear_fp32_chunk") == linear_fp32_chunk,
        f"unexpected service configuration {state}",
    )
    issues = []
    if linear_fp32_chunk is not None:
        require("linear_work" in state, "missing FP32 worker execution counts")
        check_linear_usage(state["linear_work"])
    observations = defaultdict(list)
    for obs in audit["requests"]:
        observations[obs["prompt_sha256"]].append(obs)
    planned = {r["prompt_sha256"] for r in primers + plan}
    if (
        set(observations) - planned
        or state["live_blocks"]
        or state["prefix_queries"] != len(primers) + len(plan)
    ):
        issues.append(dict(control="unexpected, missing or unreleased requests", state=state))
    if sum(o["cached_tokens"] for o in audit["requests"]) != state["prefix_hits"]:
        issues.append(dict(control="per-request hits disagree with engine hit counter"))
    for req, row in zip(plan, rows, strict=True):
        obs = observations[req["prompt_sha256"]]
        if not row["ok"] or len(obs) != 1:
            issues.append(
                dict(
                    id=req["id"],
                    control="failed HTTP or missing/duplicate worker record",
                    error=row.get("error"),
                )
            )
            continue
        actual = obs[0]
        hits = actual["cached_tokens"]
        low, high = (0, 0) if mode == "off" else (req["min_primed"], req["max_possible"])
        row.update(cached_tokens=hits, tokens=actual["tokens"])
        if not (
            low <= hits <= high
            and hits % BLOCK == 0
            and len(actual["tokens"]) == req["output"]
            and actual["exit_depths"] == [4] * req["output"]
            and actual["finish_reason"] == "length"
        ):
            issues.append(
                dict(
                    id=req["id"],
                    control="worker hit range or decode",
                    expected_hits=[low, high],
                    actual=actual,
                )
            )
    return issues


def summary(rows, wall):
    valid = [r for r in rows if r["ok"]]

    def metrics(part):
        result = {
            key: quantiles([r[key] for r in part if r.get(key) is not None])
            for key in (
                "ttft_ms",
                "first_text_ms",
                "e2e_ms",
                "queue_ms",
                "arrival_ttft_ms",
                "arrival_e2e_ms",
            )
        }
        observed = [r for r in part if "cached_tokens" in r]
        prompt_tokens = sum(r["prompt_tokens"] for r in observed)
        hits = sum(r["cached_tokens"] for r in observed)
        result.update(
            completed=len(part),
            audited=len(observed),
            hit_requests=sum(r["cached_tokens"] > 0 for r in observed),
            hit_tokens=hits,
            hit_token_pct=100 * hits / prompt_tokens if prompt_tokens else None,
        )
        return result

    return dict(
        launched=len(rows),
        errors=len(rows) - len(valid),
        wall_s=wall,
        req_s=len(valid) / wall,
        output_tok_s=sum(r["output_tokens"] for r in valid) / wall,
        overall=metrics(valid),
        cohorts={
            k: metrics([r for r in valid if r["cohort"] == k])
            for k in sorted({r["cohort"] for r in rows})
        },
    )


async def run_mode(args, report, mode, n, primers, plan, num_blocks, width):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(os.environ.get("RLT_BENCH_PORT") or sock.getsockname()[1])
    folder = report.folder / f"round-{n}-{mode}"
    folder.mkdir()
    url = f"http://127.0.0.1:{port}"
    model = "ouro-" + uuid4().hex
    audit = folder / "worker.json"
    command = [
        sys.executable,
        "-m",
        "benchmarks.ascend_main_prefix_service_worker",
        "--model",
        str(args.model.resolve()),
        "--served-model-name",
        model,
        "--device",
        "npu",
        "--dtype",
        str(benchmark_dtype()).removeprefix("torch."),
        "--attention-backend",
        "ascend_eager",
        "--single-stream",
        "--no-uva",
        "--block-size",
        str(BLOCK),
        "--num-blocks",
        str(num_blocks),
        "--max-num-seqs",
        str(args.concurrency),
        "--max-requests",
        str(max(64, args.concurrency * 2)),
        "--prefill-chunk-size",
        str(width),
        "--max-num-batched-tokens",
        str(width * args.concurrency),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if mode != "off":
        command.append("--enable-prefix-caching")
    env = dict(
        os.environ,
        RLT_BENCH_ATTENTION="grouped" if mode == "grouped" else "main",
        RLT_BENCH_AUDIT=str(audit.resolve()),
        RLT_BENCH_PREFILL_STEPS="1" if args.observe_prefill else "0",
        RLT_BENCH_LINEAR_FP32_CHUNK=str(args.linear_fp32_chunk) if args.linear_fp32_chunk else "",
    )
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        connector=aiohttp.TCPConnector(limit=args.concurrency),
    ) as client:
        async with service(
            command, folder / "server.log", env, client, url, model, args.startup_timeout
        ):
            for req in primers:
                result = await completion(client, url, model, req)
                require(result["ok"], f"primer {req['id']} failed: {result}")
            rows, wall = await burst(client, url, model, plan, args.concurrency)
            report.record("HTTP_RESPONSES", mode=mode, round=n, requests=rows)
    captured = json.loads(audit.read_text())
    issues = reconcile(
        mode, primers, plan, rows, captured, linear_fp32_chunk=args.linear_fp32_chunk
    )
    stats = summary(rows, wall)
    if args.observe_prefill:
        counts = opportunities(captured["prefill_steps"], plan)
        (folder / "opportunities.json").write_text(json.dumps(counts, indent=2) + "\n")
    stats["group_calls"] = captured["state"]["group_calls"]
    report.record(
        "SERVICE_RUN",
        mode=mode,
        round=n,
        requests=rows,
        issues=issues,
        stats=stats,
        state=captured["state"],
    )
    print(
        "SERVICE_RUN",
        mode,
        n,
        "errors",
        stats["errors"],
        "hits",
        stats["overall"]["hit_tokens"],
        "ttft_p95_ms",
        stats["overall"]["ttft_ms"]["p95"],
        "output_tok_s",
        round(stats["output_tok_s"], 2),
        flush=True,
    )
    return rows, stats, issues


def service_diffs(base, rows, label, n):
    issues = []
    for left, right in zip(base, rows, strict=True):
        require(
            left["id"] == right["id"] and left["prompt_sha256"] == right["prompt_sha256"],
            "different request traces",
        )
        if "tokens" not in left or "tokens" not in right:
            continue
        if left["tokens"] != right["tokens"] or left["text_sha256"] != right["text_sha256"]:
            pos = next(
                (i for i, (x, y) in enumerate(zip(left["tokens"], right["tokens"]), 1) if x != y),
                None,
            )
            issues.append(
                dict(
                    comparison=label,
                    round=n,
                    id=left["id"],
                    first_diff_1based=pos,
                    expected_id=left["tokens"][pos - 1] if pos else None,
                    actual_id=right["tokens"][pos - 1] if pos else None,
                    text_matches=left["text_sha256"] == right["text_sha256"],
                )
            )
    return issues


async def exercise(args, report):
    modes = ("off", "main") if args.main_only else MODES
    primers, plan = traffic(
        load_tokenizer(str(args.model.resolve())), args.requests, args.rps, args.seed
    )
    inputs = primers + plan
    # Default pool fits the entire trace even with zero physical sharing.
    blocks = args.num_blocks or sum(
        4 * math.ceil((r["prompt_tokens"] + r["output"] - 1) / BLOCK) for r in inputs
    )
    width = max(r["prompt_tokens"] + r["output"] - 1 for r in inputs)
    report.record("MANIFEST", primers=primers, requests=plan, num_blocks=blocks, block_size=BLOCK)
    manifest = report.folder / "traffic.json"
    trace = dict(primers=primers, requests=plan, num_blocks=blocks)
    manifest.write_text(json.dumps(trace, indent=2) + "\n")
    if args.reference_traffic is not None:
        expected = json.loads(args.reference_traffic.read_text())
        require(trace == expected, f"generated workload differs from {args.reference_traffic}")
    errors = []
    diffs = []
    csv_rows = []
    first = {}
    latencies = {mode: [] for mode in modes}
    grouped_calls = []
    for i in range(args.rounds):
        runs = {}
        offset = (i // 2) % len(modes)
        order = modes[offset:] + modes[:offset]
        if i % 2:
            order = order[::-1]
        for mode in order:
            report.location = dict(round=i + 1, mode=mode)
            rows, s, issues = await run_mode(
                args, report, mode, i + 1, primers, plan, blocks, width
            )
            errors.extend(dict(round=i + 1, mode=mode, **issue) for issue in issues)
            if mode in first:
                diffs.extend(service_diffs(first[mode], rows, f"{mode}_vs_first", i + 1))
            else:
                first.setdefault(mode, rows)
            runs[mode] = rows
            if mode == "grouped":
                grouped_calls.append(s["group_calls"])
            latencies[mode].append(s["overall"]["ttft_ms"]["p50"])
            for cohort, m in {"all": s["overall"], **s["cohorts"]}.items():
                csv_rows.append(
                    dict(
                        **report.math,
                        round=i + 1,
                        mode=mode,
                        cohort=cohort,
                        completed=m["completed"],
                        audited=m["audited"],
                        http_errors=s["errors"]
                        if cohort == "all"
                        else sum(not r["ok"] and r["cohort"] == cohort for r in rows),
                        hit_tokens=m["hit_tokens"],
                        hit_token_pct=m["hit_token_pct"],
                        ttft_p50_ms=m["ttft_ms"]["p50"],
                        ttft_p95_ms=m["ttft_ms"]["p95"],
                        ttft_p99_ms=m["ttft_ms"]["p99"],
                        e2e_p95_ms=m["e2e_ms"]["p95"],
                        arrival_ttft_p95_ms=m["arrival_ttft_ms"]["p95"],
                        req_s=s["req_s"] if cohort == "all" else None,
                        output_tok_s=s["output_tok_s"] if cohort == "all" else None,
                    )
                )
        for a, b in combinations(modes, 2):
            diffs.extend(service_diffs(runs[a], runs[b], f"{a}_vs_{b}", i + 1))
    write_csv(report.folder / "service.csv", csv_rows)
    report.record("SERVICE_CHECKS", control_issues=errors, output_diffs=diffs)
    paired = {}
    if not args.observe_prefill and not errors and not diffs:
        paired["off_to_main"] = savings(latencies["off"], latencies["main"])
        if grouped_calls and all(sum(counts.values()) > 0 for counts in grouped_calls):
            for mode in ("off", "main"):
                paired[f"{mode}_to_grouped"] = savings(latencies[mode], latencies["grouped"])
    return report.review(
        "FAIL" if errors or diffs else "PASS",
        requests_per_mode=args.requests,
        rounds=args.rounds,
        correctness_rounds=getattr(args, "correctness_rounds", args.rounds),
        modes=list(modes),
        diagnostic_replay=args.observe_prefill,
        control_issues=len(errors),
        correctness_status=(
            "FAIL"
            if any(
                r["round"] <= getattr(args, "correctness_rounds", args.rounds)
                for r in errors + diffs
            )
            else "PASS"
        ),
        performance_valid=not (errors or diffs),
        output_diffs=len(diffs),
        examples=(errors + diffs)[:6],
        grouped_calls_per_round=grouped_calls,
        paired_ttft=paired,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--rps",
        type=float,
        default=0,
        help="Offered Poisson arrivals/s; 0 offers a burst with bounded client concurrency",
    )
    p.add_argument("--correctness-rounds", type=int)
    p.add_argument("--rounds", type=int, default=2)
    add_precision_arg(p)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num-blocks", type=int, help="Override pool that otherwise fits all prompts")
    p.add_argument("--startup-timeout", type=float, default=240)
    p.add_argument("--out-dir", type=Path, help="New output directory")
    p.add_argument(
        "--main-only",
        action="store_true",
        help="Compare cache off and on without installing experimental grouping",
    )
    p.add_argument(
        "--reference-traffic",
        type=Path,
        help="Require the generated traffic.json to match a prior run before starting any worker",
    )
    p.add_argument(
        "--observe-prefill",
        action="store_true",
        help="Record per-step geometry in a diagnostic replay; exclude latency comparisons",
    )
    args = p.parse_args(argv)
    if args.correctness_rounds is None:
        args.correctness_rounds = args.rounds
    if not 1 <= args.correctness_rounds <= args.rounds:
        p.error("correctness rounds must fit within measured rounds")
    if (
        not args.model.is_dir()
        or args.requests < 1
        or args.rounds < 1
        or args.concurrency < 1
        or not math.isfinite(args.rps)
        or args.rps < 0
        or not math.isfinite(args.startup_timeout)
        or args.startup_timeout <= 0
        or (args.num_blocks is not None and args.num_blocks < 1)
    ):
        p.error("invalid local model, load or capacity arguments")
    report = Report("SERVICE", args.out_dir, linear_fp32_chunk=args.linear_fp32_chunk)
    return report.execute(args, lambda args, _device, report: asyncio.run(exercise(args, report)))


if __name__ == "__main__":
    raise SystemExit(main())
