"""Full parity and paired mainline prefix/grouping benefits at a single linear precision."""

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import torch

from benchmarks import ascend_main_prefix_combo as combo
from benchmarks.ascend_main_prefix_combo import ARMS, BLOCK, OUTPUTS, Case
from benchmarks.ascend_main_prefix_grouping import attention_mode, group_counts
from benchmarks.ascend_main_prefix_linear import (
    add_precision_arg,
    check_linear_usage,
    linear_usage,
    padded_fp32_linears,
)
from benchmarks.ascend_main_prefix_results import Report, ids_digest, quantiles, savings, write_csv
from benchmarks.ascend_prefix_combo_checks import require
from vllm_rlt import SamplingParams
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestPhase

MODES = ("main", "grouped")
FULL = {
    **{f"p{p}_s8": Case(p, 8) for p in (128, 256, 384, 512, 1024)},
    **{f"edge_p{p}": Case(p, 8) for p in (0, 15, 16, 17, 127, 129)},
    "repeat_p128": Case(128, 0),
    **{f"total512_p{p}": Case(p, 512 - p) for p in (0, 128, 256, 384, 512)},
    **{f"p128_s{s}": Case(128, s) for s in (1, 32, 128)},
    **{
        f"c{n}_h{h}": Case(128, 16, n, h)
        for n, h in ((1, 1), (2, 1), (2, 2), (4, 0), (4, 1), (4, 2), (4, 4))
    },
    "resident_p64": Case(64, 8, tight_pool=True),
    "evicted_p64": Case(64, 8, tight_pool=True, churn=2),
    "c4_h3": Case(128, 16, 4, 3),
}
BENEFIT = (
    "c4_h0",
    "c4_h1",
    "c4_h2",
    "c4_h3",
    "c4_h4",
    "c2_h1",
    "c2_h2",
    "c1_h1",
    "p128_s8",
    "p512_s8",
    "p1024_s8",
    "resident_p64",
    "evicted_p64",
)


def trial_order(round_index, modes=MODES):
    # Within each cache state A/B are adjacent. Rotate cache states, reverse A/B each round.
    offset = round_index % len(ARMS)
    return [
        (mode, arm)
        for arm in ARMS[offset:] + ARMS[:offset]
        for mode in (modes if (round_index + ARMS.index(arm)) % 2 == 0 else modes[::-1])
    ]


@torch.inference_mode()
def trial(model, case, mode, arm, inputs, *, linear_work=None):
    donor, miss, targets, fillers = inputs
    gc.collect()
    with attention_mode(mode):
        engine = combo.open_engine(model, case, arm)
    cache = engine.cache_manager
    starts = case.starts(arm)
    try:
        require(
            engine.execution_config.packed_prefill
            and cache.backend == "ascend_eager"
            and cache.dtype == torch.bfloat16,
            "expected packed BF16 ascend_eager",
        )
        params = SamplingParams(
            max_tokens=OUTPUTS,
            temperature=0,
            min_loops=4,
            max_loops=4,
            exit_threshold=1.0,
            ignore_eos=True,
        )
        prime = replace(params, max_tokens=1)
        engine.add_request("donor", miss if arm == "miss" else donor, prime)
        combo.drain(engine, ["donor"], 1)
        before_churn = len(cache.lookup_prefix(targets["target-0"])) * BLOCK
        require(
            before_churn == replace(case, churn=0).starts(arm)["target-0"],
            f"primer produced {before_churn} reusable tokens",
        )
        for i in range(0, len(fillers), case.count):
            ids = []
            for j, tokens in enumerate(fillers[i : i + case.count], i):
                rid = f"churn-{j}"
                engine.add_request(rid, tokens, prime)
                ids.append(rid)
            combo.drain(engine, ids, 1)
        engine.model_runner.synchronize()
        prefix_hits, queries = cache.prefix_hits, cache.prefix_queries
        kernel = cache.attention.kernel
        groups_before = group_counts(kernel)
        if linear_work is not None:
            linear_work[1].clear()
        submitted, first, finished = {}, {}, {}
        tick = time.perf_counter
        for rid, tokens in targets.items():
            submitted[rid] = tick()
            engine.add_request(rid, tokens, params)
        schedule = None
        while engine.has_unfinished_requests():
            outputs = engine.step()
            visible = tick()
            if schedule is None:
                schedule = engine.last_schedule
            for out in outputs:
                if out.token_ids:
                    first.setdefault(out.request_id, visible)
                if out.finished:
                    require(out.request_id not in finished, "duplicate finish")
                    finished[out.request_id] = visible, out
        require(
            schedule is not None and schedule.phase == RequestPhase.PREFILL,
            "unexpected first schedule",
        )
        origins = {s.request.request_id: s.request.prefill_window_origin for s in schedule.requests}
        actual = {s.request.request_id: s.token_start for s in schedule.requests}
        prefill_rows = {s.request.request_id: s.token_count for s in schedule.requests}
        require(
            origins == actual == starts, f"hit tokens {origins}, starts {actual}, expected {starts}"
        )
        require(
            cache.prefix_hits - prefix_hits == sum(starts.values())
            and cache.prefix_queries - queries == len(starts),
            "allocation counters disagree with scheduled hits",
        )
        require(
            finished.keys() == first.keys() == targets.keys() and cache.num_used_blocks == 0,
            "missing completion or live KV after finish",
        )
        end = max(t for t, _ in finished.values())
        counts = {k: v - groups_before[k] for k, v in group_counts(kernel).items()}
        hot = sum(x > 0 for x in starts.values())
        calls = cache.num_layers * cache.storage_depths
        expected = dict(
            cold=calls if mode == "grouped" and hot and case.count - hot >= 2 else 0,
            suffix=calls
            if mode == "grouped" and hot >= 2 and len(donor) - max(starts.values()) > 1
            else 0,
        )
        require(counts == expected, f"grouped calls {counts}, expected {expected}")
        rows = {}
        for rid, (t, out) in finished.items():
            require(
                out.finish_reason == "length"
                and len(out.token_ids) == OUTPUTS
                and out.exit_depths == [4] * OUTPUTS,
                f"{rid} incorrect output length, finish or exit depths",
            )
            rows[rid] = dict(
                tokens=out.token_ids,
                ttft_ms=1000 * (first[rid] - submitted[rid]),
                e2e_ms=1000 * (t - submitted[rid]),
                hit_tokens=origins[rid],
                prefill_rows=prefill_rows[rid],
            )
        work_stats = {}
        if linear_work is not None:
            work_stats["linear_work"] = linear_usage(linear_work)
            check_linear_usage(work_stats["linear_work"])
        return dict(
            requests=rows,
            wall_ms=1000 * (end - min(submitted.values())),
            before_churn=before_churn,
            hit_tokens=origins,
            group_calls=counts,
            num_blocks=cache.num_blocks,
            **work_stats,
        )
    finally:
        engine.close()


def metrics(runs):
    rows = [r for run in runs for r in run["requests"].values()]
    seconds = sum(run["wall_ms"] for run in runs) / 1000
    return dict(
        ttft_ms=quantiles([r["ttft_ms"] for r in rows]),
        e2e_ms=quantiles([r["e2e_ms"] for r in rows]),
        tpot_ms=quantiles([(r["e2e_ms"] - r["ttft_ms"]) / (len(r["tokens"]) - 1) for r in rows]),
        req_s=len(rows) / seconds,
        output_tok_s=sum(len(r["tokens"]) for r in rows) / seconds,
        hit_tokens=sum(r["hit_tokens"] for r in rows),
    )


def comparisons(rounds, modes=MODES):
    pairs = []
    if "grouped" in modes:
        pairs.extend((f"{arm}_grouping", ("main", arm), ("grouped", arm)) for arm in ARMS)
    for mode in modes:
        pairs.extend(
            (f"{mode}_{a}_to_{b}", (mode, a), (mode, b))
            for a, b in (("off", "miss"), ("miss", "reuse"), ("off", "reuse"))
        )

    def timings(key):
        return [
            median(row["ttft_ms"] for row in trial[key]["requests"].values()) for trial in rounds
        ]

    return {name: savings(timings(a), timings(b)) for name, a, b in pairs}


def run_cases(args, model, report, names, *, linear_work=None, modes=MODES):
    failed, csv_rows = [], []
    failed_comparisons = {}
    benefit_by_case = {}
    for name in names:
        print("Starting", name, flush=True)
        case = FULL[name]
        inputs = combo.workload(case, model.config.vocab_size)
        report.record(
            "INPUTS",
            case=name,
            prompts={rid: ids_digest(tokens) for rid, tokens in inputs[2].items()},
        )
        first = {}
        rounds = []
        issues = []
        for i in range(-args.warmups, args.repeats):
            burst = {}
            for mode, arm in trial_order(i, modes):
                loc = dict(case=name, round=i + 1, warmup=i < 0, mode=mode, arm=arm)
                report.location = loc
                try:
                    result = trial(model, case, mode, arm, inputs, linear_work=linear_work)
                except AssertionError as e:
                    issues.append(dict(**loc, control=str(e)))
                    report.record("CONTROL", echo=True, **issues[-1])
                    continue
                key = mode, arm
                report.record("TRIAL", **loc, **result)
                burst[key] = result
                if key in first:
                    issues.extend(
                        dict(**loc, **d)
                        for d in combo.token_diffs(first[key], result, "same_arm_first_run")
                    )
                else:
                    first[key] = result
            for mode, arm in burst:
                # Compare cache states within a mode, then compare modes within a cache state.
                checks = []
                if arm != "off":
                    checks.append((mode, "off"))
                if mode != "main":
                    checks.append(("main", arm))
                for base in checks:
                    if base in burst:
                        label = f"{base[0]}.{base[1]}_vs_{mode}.{arm}"
                        issues.extend(
                            dict(round=i + 1, warmup=i < 0, **d)
                            for d in combo.token_diffs(burst[base], burst[mode, arm], label)
                        )
            if i >= 0:
                rounds.append(burst)
        if issues:
            failed.append(name)
            for label in sorted({issue.get("comparison", "control") for issue in issues}):
                failed_comparisons.setdefault(label, []).append(name)
        summary = dict(
            case=name,
            status="FAIL" if issues else "PASS",
            issues=len(issues),
            first_issues=issues[:6],
        )
        paired = {}
        if args.experiment == "benefit" and not issues:
            paired = comparisons(rounds, modes)
            if "reuse_grouping" in paired:
                summary["reuse_grouping"] = paired["reuse_grouping"]
            benefit_by_case[name] = {
                key: paired[key]["median_saved_ms"]
                for key in (
                    "main_off_to_reuse",
                    "main_miss_to_reuse",
                    "reuse_grouping",
                    "grouped_off_to_reuse",
                    "off_grouping",
                    "miss_grouping",
                )
                if key in paired
            }
        report.record("CASE_DETAILS", case=name, all_issues=issues, paired_ttft=paired)
        report.record(report.kind + "_CASE", echo=True, **summary)
        for mode in modes:
            for arm in ARMS:
                runs = [row[mode, arm] for row in rounds if (mode, arm) in row]
                if not runs:
                    continue
                data = metrics(runs)
                csv_rows.append(
                    dict(
                        case=name,
                        **report.math,
                        status=summary["status"],
                        mode=mode,
                        arm=arm,
                        rounds=len(runs),
                        hit_tokens=data["hit_tokens"],
                        ttft_p50_ms=data["ttft_ms"]["p50"],
                        ttft_p95_ms=data["ttft_ms"]["p95"],
                        e2e_p95_ms=data["e2e_ms"]["p95"],
                        tpot_p50_ms=data["tpot_ms"]["p50"],
                        req_s=data["req_s"],
                        output_tok_s=data["output_tok_s"],
                        grouping_saved_ms=paired.get(f"{arm}_grouping", {}).get("median_saved_ms")
                        if mode == "grouped"
                        else None,
                    )
                )
    write_csv(report.folder / "cases.csv", csv_rows)
    return report.review(
        "FAIL" if failed else "PASS",
        cases=names,
        failed=failed,
        failed_comparisons=failed_comparisons,
        warmups=args.warmups,
        repeats=args.repeats,
        modes=list(modes),
        **({"paired_ttft_saved_ms": benefit_by_case} if args.experiment == "benefit" else {}),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=("bf16", "fp32", "benefit"))
    p.add_argument("--model", required=True, type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--case", choices=FULL, action="append")
    group.add_argument(
        "--all-cases", action="store_true", help="Also time all 30 cases in benefit mode"
    )
    p.add_argument("--warmups", type=int)
    p.add_argument("--repeats", type=int)
    add_precision_arg(p)
    p.add_argument("--out-dir", type=Path, help="New output directory")
    p.add_argument(
        "--main-only",
        action="store_true",
        help="Exercise the PR1 runtime without experimental grouping",
    )
    args = p.parse_args(argv)
    correctness = args.experiment != "benefit"
    if args.experiment == "fp32" and args.linear_fp32_chunk is None:
        args.linear_fp32_chunk = 256
    if args.repeats is None:
        args.repeats = 3 if correctness else 6
    if args.warmups is None:
        args.warmups = 0 if correctness else 1
    if args.repeats < 1 or args.warmups < 0 or not args.model.is_dir():
        p.error("provide a local model, positive repeats and nonnegative warmups")
    names = list(dict.fromkeys(args.case or (FULL if correctness or args.all_cases else BENEFIT)))
    kind = ("FP32" if args.linear_fp32_chunk is not None else "BF16") if correctness else "BENEFIT"
    report = Report(kind, args.out_dir, linear_fp32_chunk=args.linear_fp32_chunk)

    def run(args, device, report):
        model = OuroForCausalLM.from_pretrained(
            args.model, device=device, dtype=torch.bfloat16
        ).eval()
        report.location = {"phase": "install_linears"}
        with padded_fp32_linears(model, args.linear_fp32_chunk) as work:
            if work is not None:
                report.record("LINEAR_MODULES", modules=work[0])
            return run_cases(
                args,
                model,
                report,
                names,
                linear_work=work,
                modes=("main",) if args.main_only else MODES,
            )

    return report.execute(args, run)


if __name__ == "__main__":
    raise SystemExit(main())
