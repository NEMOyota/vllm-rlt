"""A/B test coalesced KV reads in c4_h1 using the local matrix's validated FP32 option.

Run in the checkout that supports --linear-fp32-chunk. The first burst per arm audits
actual Tensor.index_select calls on the two flat cache views during mixed prefill.
Subsequent bursts time each variant with a fresh engine, rotating their order.
"""

import argparse
import json
import math
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from unittest.mock import patch

import torch

from benchmarks import ascend_prefix_matrix as matrix
from benchmarks import ascend_prefix_workload as workload
from benchmarks.ascend_grouped_kv_cases import compare_outputs
from benchmarks.ascend_prefix_scenarios import ARMS, _mismatches
from vllm_rlt.kernels import ascend_eager
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention
from vllm_rlt.request import RequestPhase
from vllm_rlt.worker.model_runner import ModelRunner


def merged_reads(attention, q, keys, values, tables, groups, *, flat_rows, result=None):
    """Read the same physical slots, then keep the arithmetic of _forward per group."""
    flat_keys, flat_values, rows_per_block = flat_rows
    block_size, kv_heads, _ = keys.shape[-3:]
    slots = []
    for first, _, positions, _, _, _ in groups:
        blocks = tables[first, positions // block_size]
        rows = blocks.to(torch.int64) * rows_per_block + positions % block_size
        slots.append(rows.clamp(0, flat_keys.shape[0] - 1))
    lengths = [s.numel() for s in slots]
    slots = torch.cat(slots)
    key_groups = flat_keys.index_select(0, slots).split(lengths)
    value_groups = flat_values.index_select(0, slots).split(lengths)

    if result is None:
        result = torch.zeros_like(q)
    head_groups = q.shape[1] // kv_heads
    for group, k, v in zip(groups, key_groups, value_groups, strict=True):
        _, indices, _, mask, initial, _ = group
        if head_groups != 1:
            k = k.repeat_interleave(head_groups, dim=1)
            v = v.repeat_interleave(head_groups, dim=1)
        query = q.index_select(0, indices).transpose(0, 1).unsqueeze(0).contiguous()
        key = k.transpose(0, 1).unsqueeze(0)
        value = v.transpose(0, 1).unsqueeze(0)
        if not initial:
            key, value = key.contiguous(), value.contiguous()
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        scores = scores + mask[None, None, :, :]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, value).squeeze(0).transpose(0, 1)
        result.index_copy_(0, indices, output)
    return result


def new_audit():
    return dict(
        steps=Counter(),
        prefill_spans=[],
        mixed_forwards=0,
        groups=0,
        merged_by_phase=Counter(),
        kv_index_select=Counter(),
    )


@contextmanager
def route(merged, audit):
    """Limit the hook to the target burst and to a mixed-hit atomic prefill submission."""
    execute = ModelRunner._execute
    original = AscendEagerAttention._forward
    index_select = torch.Tensor.index_select

    def run(runner, schedule, prepared):
        phase = schedule.phase if schedule is not None else None
        label = phase.value if phase is not None else "mixed_or_unscheduled"
        if audit is not None:
            audit["steps"][label] += 1
        if phase != RequestPhase.PREFILL:
            return execute(runner, schedule, prepared)
        spans = schedule.requests
        if audit is not None:
            audit["prefill_spans"].append(
                [[s.request.request_id, s.token_start, s.token_count] for s in spans]
            )
        if not (
            any(s.token_start == 0 for s in spans)
            and any(
                s.token_start > 0 and s.token_start == s.request.prefill_window_origin
                for s in spans
            )
            and all(s.token_count > 1 for s in spans)
        ):
            return execute(runner, schedule, prepared)

        def forward(attention, q, k, v, tables, groups, *, flat_rows=None, result=None):
            call = original
            eligible = (
                flat_rows is not None
                and len(groups) == len(spans)
                and all(g[1].numel() > 1 for g in groups)
                and any(g[4] for g in groups)
                and not all(g[4] for g in groups)
            )
            if eligible and merged:
                call = merged_reads
            counter = nullcontext()
            if eligible and audit is not None:
                audit["mixed_forwards"] += 1
                audit["groups"] += len(groups)
                if merged:
                    audit["merged_by_phase"][label] += 1

                def count_select(tensor, dim, index):
                    if tensor is flat_rows[0]:
                        audit["kv_index_select"]["k"] += 1
                    if tensor is flat_rows[1]:
                        audit["kv_index_select"]["v"] += 1
                    return index_select(tensor, dim, index)

                counter = patch.object(torch.Tensor, "index_select", count_select)
            with counter:
                return call(attention, q, k, v, tables, groups, flat_rows=flat_rows, result=result)

        with patch.object(AscendEagerAttention, "_forward", forward):
            return execute(runner, schedule, prepared)

    with (
        patch.object(ModelRunner, "_execute", run) if merged or audit is not None else nullcontext()
    ):
        yield


def audit_issues(audits, expected_forwards):
    issues = []
    for mode, arms in audits.items():
        for arm, audit in arms.items():
            active = expected_forwards if arm == "reuse" else 0
            reads = active * (1 if mode == "merged" else 4)
            spans = [
                [
                    f"target-{i}",
                    128 if arm == "reuse" and i == 0 else 0,
                    16 if arm == "reuse" and i == 0 else 144,
                ]
                for i in range(4)
            ]
            checks = {
                "prefill_spans": (audit["prefill_spans"], [spans]),
                "observed_decode": (any(p != "prefill" for p in audit["steps"]), True),
                "mixed_forwards": (audit["mixed_forwards"], active),
                "groups": (audit["groups"], active * 4),
                "merged_by_phase": (
                    dict(audit["merged_by_phase"]),
                    {"prefill": active} if mode == "merged" and active else {},
                ),
                "kv_index_select": (
                    dict(audit["kv_index_select"]),
                    {"k": reads, "v": reads} if reads else {},
                ),
            }
            issues.extend(
                dict(mode=mode, arm=arm, check=key, actual=value, expected=expected)
                for key, (value, expected) in checks.items()
                if value != expected
            )
    return issues


def token_diffs(reference, actual):
    try:
        return [
            dict(target=f"target-{i}", first_diff_1based=pos, expected=a, actual=b)
            for i, pos, a, b in compare_outputs(reference, actual)
        ]
    except ValueError as exc:
        return [dict(error=str(exc))]


def paired_ms(original, changed, candidate, baseline="original"):
    saved = [a - b for a, b in zip(original, changed, strict=True)]
    return {
        f"{baseline}_ms": round(median(original), 2),
        f"{candidate}_ms": round(median(changed), 2),
        "paired_saved_ms": [round(x, 2) for x in saved],
        "median_paired_saved_ms": round(median(saved), 2),
    }


def ttft_review(measured, candidate="merged", baseline="original"):
    # Four requests in one burst are correlated. Pair variants within each repeat.
    times = {
        arm: {
            mode: [median(r["ttft_ms"] for r in pair[mode][arm]["requests"]) for pair in measured]
            for mode in (baseline, candidate)
        }
        for arm in ARMS
    }
    times["reuse_minus_miss"] = {
        mode: [r - m for r, m in zip(times["reuse"][mode], times["miss"][mode], strict=True)]
        for mode in (baseline, candidate)
    }
    return {
        arm: paired_ms(p50[baseline], p50[candidate], candidate, baseline)
        for arm, p50 in times.items()
    }


def timing_comparisons(runs, modes):
    measured = runs[1:]
    comparisons = {}
    for base, candidate in combinations(modes, 2):
        summary = ttft_review(measured, candidate, base)
        requests = {}
        for i, row in enumerate(runs[0][base]["reuse"]["requests"]):
            base_times = [r[base]["reuse"]["requests"][i]["ttft_ms"] for r in measured]
            times = [r[candidate]["reuse"]["requests"][i]["ttft_ms"] for r in measured]
            requests[f"target-{i}"] = dict(
                hit_tokens=row["hit_tokens"],
                **paired_ms(base_times, times, candidate, base),
            )
        comparisons[f"{base}_vs_{candidate}"] = dict(ttft=summary, reuse_requests=requests)
    return {"comparisons": comparisons}


class GateFailure(RuntimeError):
    """A completed set of six arms failed its output or path checks."""


class Pairs:
    case_name = "c4_h1"
    intermediate_modes = ()
    label = "GATHER"
    candidate = "merged"
    audit_scope = "target mixed-prefill flat K/V Tensor.index_select calls"
    route = staticmethod(route)
    new_audit = staticmethod(new_audit)
    audit_issues = staticmethod(audit_issues)
    profile_pair = None
    stock_methods = ("attend_metadata", "_forward")

    def __init__(
        self, run_trial, stream, reference, *, repeats=3, trace_dir=None, case_name=None, modes=None
    ):
        self.run_trial = run_trial
        self.stream = stream
        self.reference = reference
        self.runs = []
        self.counts = Counter()
        self.case_name = case_name or self.case_name
        self.modes = modes or ("original", *self.intermediate_modes, self.candidate)
        self.candidate = self.modes[-1]
        self.audits = {mode: {} for mode in self.modes}
        self.repeats = repeats
        self.trace_dir = trace_dir
        self.traces = {}

    def record(self, kind, **fields):
        print(json.dumps(dict(kind=kind, **fields), allow_nan=False), file=self.stream, flush=True)

    def variant_route(self, mode, audit):
        return self.route(mode == self.candidate, audit)

    def run_variant(self, mode, audit, trial_args, trial_kwargs, profile_target=None):
        burst = workload._target_burst

        def target(engine, targets, params, expected):
            profile = profile_target(engine, targets) if profile_target else nullcontext()
            with profile, self.variant_route(mode, audit):
                return burst(engine, targets, params, expected)

        with patch.object(workload, "_target_burst", target):
            return self.run_trial(*trial_args, **trial_kwargs)

    def timing_summary(self):
        return dict(ttft=ttft_review(self.runs[1:], self.candidate))

    def __call__(self, model, backend, case, arm, output_tokens, seed, *extra, **kwargs):
        if not self.runs:
            source = Path(ascend_eager.__file__).resolve()
            for name in self.stock_methods:
                method = getattr(AscendEagerAttention, name)
                if Path(method.__code__.co_filename).resolve() != source:
                    raise RuntimeError(f"disable the other attention hook before A/B: {method}")
            self.record(
                "model",
                backend=backend,
                decoder_layers=model.config.num_hidden_layers,
                loops=model.config.total_ut_steps,
                torch_version=str(torch.__version__),
                matrix_source=str(Path(matrix.__file__).resolve()),
            )
        repeat = self.counts[arm]
        self.counts[arm] += 1
        if repeat == len(self.runs):
            self.runs.append({mode: {} for mode in self.modes})
        pair = self.runs[repeat]
        offset = (repeat + ARMS.index(arm)) % len(self.modes)
        order = self.modes[offset:] + self.modes[:offset]
        for mode in order:
            audit = self.new_audit() if repeat == 0 else None
            trial = self.run_variant(
                mode, audit, (model, backend, case, arm, output_tokens, seed, *extra), kwargs
            )
            pair[mode][arm] = trial
            if audit is not None:
                self.audits[mode][arm] = audit
            self.record(
                "arm",
                repeat=repeat,
                audit_warmup=repeat == 0,
                arm=arm,
                mode=mode,
                order=order,
                requests=[
                    {"request_id": f"target-{i}", **row} for i, row in enumerate(trial["requests"])
                ],
                audit=audit,
                prefix_hits=trial["prefix_hits"],
                prefix_queries=trial["prefix_queries"],
            )

        if len(pair[self.candidate]) == len(ARMS):
            issues = []
            if self.reference is None:
                self.reference = pair["original"]["off"]
            for mode, arms in pair.items():
                issues.extend(dict(mode=mode, **m) for m in _mismatches(case, [arms]))
                for a in ARMS:
                    for diff in token_diffs(self.reference, arms[a]):
                        issues.append(dict(mode=mode, arm=a, check="vs_reference_off", **diff))
            for a in ARMS:
                for mode in self.modes[1:]:
                    for diff in token_diffs(pair["original"][a], pair[mode][a]):
                        issues.append(dict(mode=mode, arm=a, check=f"original_vs_{mode}", **diff))
            if repeat == 0:
                calls = model.config.num_hidden_layers * model.config.total_ut_steps
                issues.extend(self.audit_issues(self.audits, calls))
                print(f"{self.label}_AUDIT", json.dumps(self.audits), flush=True)
            self.record("gate", repeat=repeat, passed=not issues, issues=issues)
            print(
                f"{self.label}_GATE",
                json.dumps(dict(repeat=repeat, passed=not issues, issues=issues)),
                flush=True,
            )
            if issues:
                raise GateFailure(
                    f"{self.case_name} gate failed on repeat {repeat}; see {self.label}_GATE"
                )
            if self.trace_dir is not None and repeat == self.repeats:
                # The server's all-Linear patch is still active here. Keep trace replays
                # out of timing samples and retain the measurements if trace export fails.
                timing = self.timing_summary()
                self.record("timing", measured_pairs=repeat, **timing)
                print(f"{self.label}_TIMING", json.dumps(timing), flush=True)
                self.traces = self.profile_pair(
                    (model, backend, case, "reuse", output_tokens, seed, *extra), kwargs
                )
        return pair["original"][arm]


def reference_settings(path, case_name="c4_h1"):
    if path is None:
        return None, None
    report = json.loads(path.read_text())
    case = report["cases"][case_name]
    if report["state"] != "complete" or case["summary"]["status"] != "PASS":
        raise ValueError(f"reference must contain a completed, passing {case_name} matrix run")
    rows = None
    for section in (report.get("contract", {}), report.get("metadata", {}), report):
        for key in (
            "linear_fp32_chunk",
            "linear_fp32_chunk_rows",
            "fp32_chunk_rows",
            "linear_fp32",
        ):
            value = section.get(key)
            if isinstance(value, dict):
                value = value.get("chunk_rows")
            if type(value) is int and value > 0:
                rows = value
                break
        if rows is not None:
            break
    return rows, case["trials"][0]["off"]


def argument_parser(pair_type, description, *, reference_required=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--reference", type=Path, required=reference_required, help="accepted full FP32 matrix JSON"
    )
    parser.add_argument("--chunk-rows", type=int, help="same value as the accepted FP32 run")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out-dir", type=Path, help="new directory (default: results/<experiment>-<case>-UTC)"
    )
    if pair_type.profile_pair is not None:
        parser.add_argument(
            "--trace-reuse",
            action="store_true",
            help="after passing timing gates, trace one fresh reuse burst per variant",
        )
    return parser


def main(argv=None, *, pair_type=Pairs, description=__doc__):
    parser = argument_parser(pair_type, description)
    return run_experiment(parser, parser.parse_args(argv), pair_type)


def run_experiment(parser, args, pair_type, *, case_name=None, modes=None):
    tag = pair_type.label.lower()
    case_name = case_name or pair_type.case_name
    saved_rows, reference = reference_settings(args.reference, case_name)
    rows = args.chunk_rows if args.chunk_rows is not None else saved_rows
    if rows is None or rows < 1:
        parser.error("supply --chunk-rows with the --linear-fp32-chunk value of the accepted run")
    if saved_rows is not None and rows != saved_rows:
        parser.error(f"reference was produced with --linear-fp32-chunk {saved_rows}")
    if not args.model.is_dir() or args.repeats < 1:
        parser.error("provide a local model directory and at least one measured repeat")
    out = args.out_dir or Path("results") / datetime.now(timezone.utc).strftime(
        f"{tag}-{case_name}-%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    report = out / f"{tag}.jsonl"
    print("RESULTS", report.resolve(), flush=True)
    with report.open("x", encoding="utf-8") as stream:
        pairs = pair_type(
            matrix.run_trial,
            stream,
            reference,
            repeats=args.repeats,
            case_name=case_name,
            modes=modes,
            trace_dir=out / "traces" if getattr(args, "trace_reuse", False) else None,
        )
        pairs.record(
            "config",
            case=case_name,
            precision="local matrix --linear-fp32-chunk",
            chunk_rows=rows,
            repeats=args.repeats,
            seed=168,
            output_tokens=16,
            candidate=pairs.candidate,
            modes=pairs.modes,
            audit_scope=pairs.audit_scope,
            ttft_definition="burst median (ms), paired_saved_ms = baseline - candidate",
            reference=str(args.reference) if args.reference else "live original/off",
        )
        try:
            with patch.object(matrix, "run_trial", pairs):
                status = matrix.main(
                    [
                        "--model",
                        str(args.model),
                        "--device",
                        "npu",
                        "--case",
                        case_name,
                        "--linear-fp32-chunk",
                        str(rows),
                        "--output-tokens",
                        "16",
                        "--seed",
                        "168",
                        "--warmups",
                        "1",
                        "--repeats",
                        str(args.repeats),
                        "--out",
                        str(out / "original-matrix.json"),
                    ]
                )
            if status or pairs.counts != {arm: args.repeats + 1 for arm in ARMS}:
                raise RuntimeError(
                    f"incomplete matrix replay: status={status}, arms={pairs.counts}"
                )
            review = dict(
                status="PASS",
                case=case_name,
                measured_pairs=args.repeats,
                **pairs.timing_summary(),
            )
            if pairs.traces:
                review["reuse_traces"] = pairs.traces
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            review = dict(
                status="FAIL" if isinstance(exc, GateFailure) else "ERROR", error=repr(exc)
            )
            if isinstance(exc, SystemExit):
                review["matrix_cli"] = "Use the server checkout providing --linear-fp32-chunk."
            pairs.record("review", **review)
            print(f"{pairs.label}_REVIEW", json.dumps(review), flush=True)
            raise
        pairs.record("review", **review)
        print(f"{pairs.label}_REVIEW", json.dumps(review), flush=True)


if __name__ == "__main__":
    main()
