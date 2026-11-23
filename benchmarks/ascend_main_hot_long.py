"""Replay one real 242-token HTTP prompt and its 256/257-row controls on one Ascend engine."""

import argparse
import gc
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks import ascend_main_prefix_matrix as matrix
from benchmarks.ascend_main_hot_long_audit import PrefillOps
from benchmarks.ascend_main_prefix_linear import (
    PreparedFP32Linears,
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
)
from benchmarks.ascend_main_prefix_results import Report, ids_digest, quantiles, savings, write_csv
from benchmarks.ascend_main_prefix_service import shared_tokens
from benchmarks.ascend_prefix_combo_checks import require
from vllm_rlt import SamplingParams
from vllm_rlt.models import OuroForCausalLM, load_tokenizer
from vllm_rlt.request import RequestPhase

HIT = 208
CHUNK = 256
LENGTHS = (242, 256, 257)
ARMS = combo.ARMS


def arm_order(index):
    # Three rotations and their reverses balance each position over six measured rounds.
    rotation = (index // 2) % len(ARMS)
    order = ARMS[rotation:] + ARMS[:rotation]
    return order if index % 2 == 0 else order[::-1]


def service_input(path, tokenizer, request_id):
    with path.open(encoding="utf-8-sig") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    manifests = [r for r in records if r["kind"] == "MANIFEST"]
    require(len(manifests) == 1, "expected one service MANIFEST in the JSONL")
    manifest = manifests[0]
    candidates = [
        r
        for r in manifest["requests"]
        if r["cohort"] == "hot_long"
        and r["prompt_tokens"] == LENGTHS[0]
        and r["min_primed"] == HIT
        and (request_id is None or r["id"] == request_id)
    ]
    require(bool(candidates), "no matching 242-token hot_long request with 208 primed tokens")
    target = candidates[0]
    primers = [r for r in manifest["primers"] if r["id"] == "primer_long"]
    require(len(primers) == 1, "expected the service's primer_long")

    def encode(row):
        ids = list(tokenizer.encode(row["prompt"]))
        require(
            len(ids) == row["prompt_tokens"] and ids_digest(ids) == row["prompt_sha256"],
            f"tokenizer disagrees with archived token IDs for {row['id']}",
        )
        return ids

    base, donor = encode(target), encode(primers[0])
    require(shared_tokens(base, donor) == HIT, "archived primer does not yield 208 complete tokens")
    first = set()
    for run in (r for r in records if r["kind"] == "SERVICE_RUN"):
        observed = [r for r in run["requests"] if r["id"] == target["id"]]
        require(len(observed) == 1, "service replay has a missing or duplicated target")
        row = observed[0]
        require(
            row["ok"]
            and row["prompt_sha256"] == target["prompt_sha256"]
            and row["tokens"]
            and row["cached_tokens"] == (0 if run["mode"] == "off" else HIT),
            "archived service target failed its identity or hit check",
        )
        first.add(row["tokens"][0])
    require(len(first) == 1, "expected a single consistent archived first token")
    # Extend token IDs directly so 242 is unchanged and 257 is exactly 256 plus one ID.
    extended = base + base[-(LENGTHS[-1] - len(base)) :]
    return {n: extended[:n] for n in LENGTHS}, donor, target["id"], first.pop()


@torch.inference_mode()
def trial(
    model, target, primer, arm, work, diagnostic=False, *, chunk_rows=CHUNK, time_q_pad=False
):
    gc.collect()
    # Use exactly the same cache capacity and scheduler limits at all three lengths.
    engine = combo.open_engine(model, combo.Case(HIT, LENGTHS[-1] - HIT), arm)
    cache = engine.cache_manager
    try:
        require(
            engine.execution_config.packed_prefill
            and cache.backend == "ascend_eager"
            and cache.dtype == torch.bfloat16
            and cache.enable_prefix_caching == (arm != "off"),
            "wrong mainline config",
        )
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            min_loops=4,
            max_loops=4,
            exit_threshold=1.0,
            ignore_eos=True,
        )
        engine.add_request("primer", primer, params)
        combo.drain(engine, ["primer"], 1)
        engine.model_runner.synchronize()
        n = len(target)
        hit = HIT if arm == "reuse" else 0
        before_hits, before_queries = cache.prefix_hits, cache.prefix_queries
        kernel = cache.attention.kernel
        before_batched = kernel.batched_prefill_calls
        work[1].clear()
        ops = (
            PrefillOps(cache.device, chunk_rows, n, hit, time_q_pad=time_q_pad)
            if diagnostic
            else None
        )
        schedule = None
        with ops.observe(model) if ops is not None else nullcontext():
            start = time.perf_counter_ns()
            engine.add_request("target", target, params)
            while engine.has_unfinished_requests():
                outputs = engine.step()
                visible = time.perf_counter_ns()
                if schedule is None:
                    schedule = engine.last_schedule
                if outputs and outputs[0].token_ids:
                    break
            else:
                raise AssertionError("target finished without a visible token")
        require(
            len(outputs) == 1
            and outputs[0].request_id == "target"
            and outputs[0].finished
            and outputs[0].exit_depths == [4]
            and len(outputs[0].token_ids) == 1
            and outputs[0].finish_reason == "length",
            "invalid target completion",
        )
        require(
            schedule.phase == RequestPhase.PREFILL and len(schedule.requests) == 1,
            "expected one ordinary target prefill",
        )
        scheduled = schedule.requests[0]
        require(
            scheduled.token_start == scheduled.request.prefill_window_origin == hit
            and schedule.num_tokens == n - hit
            and cache.prefix_hits - before_hits == hit
            and cache.prefix_queries - before_queries == 1
            and not engine.has_unfinished_requests()
            and cache.num_used_blocks == 0,
            f"unexpected target origin, query rows, counters or unreleased KV (expected {hit=})",
        )
        expected_calls = cache.num_layers * cache.storage_depths
        batched = kernel.batched_prefill_calls - before_batched
        require(batched == (0 if hit else expected_calls), "unexpected packed attention path")
        usage = linear_usage(work)
        check_linear_usage(usage)
        result = dict(
            first_token=outputs[0].token_ids[0],
            hit_tokens=hit,
            prefill_rows=schedule.num_tokens,
            batched_prefill_calls=batched,
            linear_work=usage,
        )
        if ops is None:
            result["ttft_ms"] = (visible - start) / 1e6
        else:
            result["ops"] = ops.result(expected_calls)
        return result
    finally:
        engine.close()


def target_schedule(args):
    for phase, rounds in (("warmup", 1), ("timing", args.repeats), ("audit", args.audit_repeats)):
        for i in range(rounds):
            offset = i % len(LENGTHS)
            for n in LENGTHS[offset:] + LENGTHS[:offset]:
                order_index = 2 * i if phase == "audit" else i
                for arm in arm_order(order_index + 2 * LENGTHS.index(n)):
                    yield phase, i, n, arm
            print(f"HOT_LONG_PROGRESS {phase} {i + 1}/{rounds}", flush=True)


def summarize_chunk(n, archive_token, records, issues):
    tokens = sorted({record["first_token"] for record in records})
    if len(tokens) != 1:
        issues.append(dict(reason="first-token parity across arms or replays", tokens=tokens))
    archive_match = tokens == [archive_token] if n == LENGTHS[0] else None
    if archive_match is False:
        issues.append(
            dict(
                reason="242-token first output differs from archived HTTP",
                expected=archive_token,
                actual=tokens,
            )
        )
    times, audits = {}, {}
    for arm in ARMS:
        runs = [r for r in records if r["arm"] == arm]
        times[arm] = [r["ttft_ms"] for r in runs if r["phase"] == "timing"]
        audits[arm] = [r["ops"] for r in runs if r["phase"] == "audit"]
        counts = {(r["linear_tiles"], r["q_proj_tiles"]) for r in audits[arm]}
        if len(counts) > 1:
            issues.append(dict(reason="linear tile count changed between audits", arm=arm))
    arms, paired, paired_ops = {}, {}, {}
    if not issues:
        for arm in ARMS:
            samples = audits[arm]
            first = samples[0]
            dist = quantiles(times[arm])
            hit = HIT if arm == "reuse" else 0
            arms[arm] = dict(
                ttft_p50_ms=dist["p50"],
                ttft_p95_ms=dist["p95"],
                hit_tokens=hit,
                prefill_rows=n - hit,
                clock=first["clock"],
                linear_tiles=first["linear_tiles"],
                linear_padded_rows=first["linear_padded_rows"],
                q_proj_tiles=first["q_proj_tiles"],
                kv_read_p50_ms=round(median(s["kv_read_ms"] for s in samples), 3),
                sdpa_p50_ms=round(median(s["sdpa_ms"] for s in samples), 3),
            )
            if "q_pad" in first["op_ms"]:
                arms[arm]["q_pad_p50_ms"] = round(median(s["op_ms"]["q_pad"] for s in samples), 3)
        for left, right in (("off", "miss"), ("off", "reuse"), ("miss", "reuse")):
            paired[f"{left}_to_{right}"] = savings(times[left], times[right])
        for key in ("kv_read_ms", "sdpa_ms"):
            paired_ops[key] = savings(
                [r[key] for r in audits["miss"]], [r[key] for r in audits["reuse"]]
            )
    return dict(
        status="FAIL" if issues else "PASS",
        first_tokens=tokens,
        archived_first_match=archive_match,
        arms=arms,
        paired_ttft=paired,
        audit_miss_to_reuse=paired_ops,
        issues=issues,
    ), times


def check_sequences(model, n, target, donor, miss, chunks, prepared, records, report):
    # Later tokens were sensitive to Linear geometry in previous experiments.
    # Run the existing 16-token matrix oracle separately from both timing phases.
    case = combo.Case(HIT, n - HIT)
    inputs = donor, miss, {"target-0": target}, []
    results, issues = {}, []
    for chunk in chunks:
        with prepared.fixed_rows(chunk) as work:
            for arm in ARMS:
                where = dict(phase="parity16", length=n, linear_fp32_chunk=chunk, arm=arm)
                report.location = where
                try:
                    result = matrix.trial(model, case, "main", arm, inputs, linear_work=work)
                except AssertionError as error:
                    issues.append(dict(**where, reason=str(error)))
                    report.record("CONTROL", echo=True, **issues[-1])
                    continue
                results[chunk, arm] = result
                ids = result["requests"]["target-0"]["tokens"]
                seen = {r["first_token"] for r in records[n, chunk] if r["arm"] == arm}
                if seen != {ids[0]}:
                    issues.append(
                        dict(
                            **where,
                            reason="one-token and 16-token replay disagree",
                            short=sorted(seen),
                            long_first=ids[0],
                        )
                    )
                report.record("TOKENS16", **where, tokens=ids, hit_tokens=result["hit_tokens"])
    reference = results.get((CHUNK, "off"))
    if reference is not None:
        for (chunk, arm), result in results.items():
            issues.extend(combo.token_diffs(reference, result, f"r{CHUNK}.off_vs_r{chunk}.{arm}"))
    return dict(status="FAIL" if issues else "PASS", output_tokens=combo.OUTPUTS, issues=issues)


def run_cases(model, targets, donor, archive_token, args, report, work=None):
    miss = [(donor[0] + 1) % model.config.vocab_size, *donor[1:]]
    require(shared_tokens(miss, targets[LENGTHS[0]]) == 0, "miss primer unexpectedly shares a page")
    report.record(
        "INPUTS",
        archived_first=archive_token,
        target_sha256={n: ids_digest(t) for n, t in targets.items()},
        donor_sha256=ids_digest(donor),
        miss_donor_sha256=ids_digest(miss),
    )
    candidate = getattr(args, "compare_chunk", None)
    chunks = (CHUNK, candidate) if candidate is not None else (CHUNK,)
    prepared = PreparedFP32Linears(model) if candidate is not None else None
    records, issues = defaultdict(list), defaultdict(list)
    for phase, i, n, arm in target_schedule(args):
        # Compare adjacent trials of the same arm. Alternate which size goes first.
        reverse = (i + LENGTHS.index(n) + ARMS.index(arm)) % 2
        order = chunks[::-1] if reverse else chunks
        for rows in order:
            where = dict(phase=phase, round=i + 1, length=n, arm=arm, linear_fp32_chunk=rows)
            report.location = where
            scope = prepared.fixed_rows(rows) if prepared is not None else nullcontext(work)
            with scope as active:
                try:
                    result = trial(
                        model,
                        targets[n],
                        miss if arm == "miss" else donor,
                        arm,
                        active,
                        diagnostic=phase == "audit",
                        chunk_rows=rows,
                        time_q_pad=prepared is not None,
                    )
                except AssertionError as exc:
                    problem = dict(**where, reason=str(exc))
                    issues[n, rows].append(problem)
                    report.record("CONTROL", echo=True, **problem)
                    continue
            record = where | result
            report.record("TARGET", **record)
            records[n, rows].append(record)

    cases, csv_rows = {}, []
    for n in LENGTHS:
        by_chunk, times = {}, {}
        for rows in chunks:
            summary, times[rows] = summarize_chunk(
                n, archive_token, records[n, rows], issues[n, rows]
            )
            by_chunk[rows] = summary
        if prepared is None:
            cases[n] = by_chunk[CHUNK]
        else:
            sequence = check_sequences(
                model, n, targets[n], donor, miss, chunks, prepared, records, report
            )
            valid = sequence["status"] == "PASS" and all(
                v["status"] == "PASS" for v in by_chunk.values()
            )
            across = (
                {arm: savings(times[CHUNK][arm], times[candidate][arm]) for arm in ARMS}
                if valid
                else {}
            )
            cases[n] = dict(
                status="PASS" if valid else "FAIL",
                by_chunk=by_chunk,
                parity16=sequence,
                paired_256_to_candidate=across,
            )
        for rows, summary in by_chunk.items():
            for arm, metrics in summary["arms"].items():
                csv_rows.append(
                    dict(
                        length=n,
                        arm=arm,
                        linear_fp32_chunk=rows,
                        status=cases[n]["status"],
                        **metrics,
                    )
                )
        report.record("HOT_LONG_CASE", echo=True, length=n, **cases[n])
    write_csv(report.folder / "cases.csv", csv_rows)
    failed = [n for n, v in cases.items() if v["status"] == "FAIL"]
    return report.review(
        "FAIL" if failed else "PASS",
        cases=cases,
        failed=failed,
        repeats=args.repeats,
        audit_repeats=args.audit_repeats,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--service-raw", type=Path, required=True, help="Night run paced/raw.jsonl")
    p.add_argument(
        "--compare-chunk",
        type=int,
        metavar="ROWS",
        help="Pair a fixed FP32 tile size with the 256-row control",
    )
    p.add_argument("--request-id", help="Default: first matching 242-token hot_long request")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--audit-repeats", type=int, default=3)
    p.add_argument(
        "--out-dir", type=Path, help="New directory for raw.jsonl, cases.csv and review.json"
    )
    args = p.parse_args(argv)
    if (
        not args.model.is_dir()
        or not args.service_raw.is_file()
        or min(args.repeats, args.audit_repeats) < 1
    ):
        p.error("provide a local model, archived service raw.jsonl and positive repeat counts")
    if args.compare_chunk is not None and (args.compare_chunk < 1 or args.compare_chunk == CHUNK):
        p.error("--compare-chunk must be a positive row count different from 256")
    report = Report(
        "HOT_LONG", args.out_dir, linear_fp32_chunk=CHUNK, compare_chunk=args.compare_chunk
    )

    def run(args, device, report):
        report.location = {"phase": "restore_prompt"}
        tokenizer = load_tokenizer(str(args.model.resolve()))
        targets, donor, rid, archived = service_input(args.service_raw, tokenizer, args.request_id)
        report.record(
            "SOURCE",
            echo=True,
            service_raw=str(args.service_raw.resolve()),
            request_id=rid,
            base_sha256=ids_digest(targets[LENGTHS[0]]),
            archived_first=archived,
            sequence_check_tokens=combo.OUTPUTS if args.compare_chunk is not None else 1,
        )
        report.location = {"phase": "load_model"}
        model = OuroForCausalLM.from_pretrained(
            args.model, device=device, dtype=torch.bfloat16
        ).eval()
        if args.compare_chunk is not None:
            return run_cases(model, targets, donor, archived, args, report)
        with padded_fp32_linears(model, CHUNK) as work:
            return run_cases(model, targets, donor, archived, args, report, work)

    return report.execute(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
