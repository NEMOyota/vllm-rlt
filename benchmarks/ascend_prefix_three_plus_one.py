"""Compare three batched cold prompts and one cached suffix against native c4_h1 attention.

Run with the matrix checkout that produced the accepted all-Linear FP32 report.
--trace-reuse adds two fresh replays after all paired measurements pass.
"""

import json
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_scenarios import _mismatches
from benchmarks.ascend_prefix_trace import npu_profiler, trace_target
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention, _prefill_geometry
from vllm_rlt.request import RequestPhase
from vllm_rlt.worker.model_runner import ModelRunner

if __package__:
    from . import ascend_prefix_mixed_gather as paired
else:  # Both scripts can be exported side by side, outside the server's patched checkout.
    import ascend_prefix_mixed_gather as paired


@dataclass
class ColdBatch:
    """A private write-slot/mask cache for equal-length complete prompts at one loop depth."""

    cold_rows: slice | torch.Tensor
    hot_groups: list
    cold_count: int
    width: int
    write_blocks: torch.Tensor
    write_offsets: torch.Tensor
    kernel: AscendEagerAttention = field(default_factory=AscendEagerAttention)

    @classmethod
    def plan(cls, batch, groups, cold_count=3):
        cold = [g for g in groups if g[4]]  # _plan already verified positions 0..width-1.
        hot = [g for g in groups if not g[4]]
        if (
            len(cold) != cold_count
            or len(hot) != 4 - cold_count
            or any(g[1].numel() <= 1 for g in hot)
        ):
            return None
        width = cold[0][1].numel()
        if width <= 1 or any(g[1].numel() != width for g in cold):
            return None
        if len({batch.rows[g[0]][1] for g in groups}) != 1 or any(
            batch.rows[g[0]][2] <= 0 for g in hot
        ):
            return None

        start = cold[0][0]
        cold_rows = slice(start, start + cold_count * width)
        if _prefill_geometry(batch.rows[cold_rows]) != (cold_count, width):
            # A hot request can sit between cold requests. Pack the cold groups in
            # their original order and scatter by their actual query indices.
            cold_rows = torch.cat([g[1] for g in cold])
        return cls(
            cold_rows,
            hot,
            cold_count,
            width,
            batch.write_blocks[cold_rows],
            batch.write_offsets[cold_rows],
        )

    def suffixes(self, native, owner, q, keys, values, tables, flat_rows, result):
        return native(
            owner, q, keys, values, tables, self.hot_groups, flat_rows=flat_rows, result=result
        )

    def apply(self, native, owner, q, keys, values, tables, flat_rows, result):
        result = self.suffixes(native, owner, q, keys, values, tables, flat_rows, result)
        cold = self.kernel._batched_prefill(
            q[self.cold_rows], keys, values, self, flat_rows, (self.cold_count, self.width)
        )
        if isinstance(self.cold_rows, slice):
            result[self.cold_rows].copy_(cold)
        else:
            result.index_copy_(0, self.cold_rows, cold)
        return result


def split_forward(native, audit, plan_batch=ColdBatch.plan):
    last_batch = last_layout = plan = None

    def forward(owner, q, keys, values, tables, groups, *, flat_rows=None, result=None):
        nonlocal last_batch, last_layout, plan
        batch = owner._metadata() if owner._metadata is not None else None
        if batch is not None and flat_rows is not None and len(groups) == 4:
            layout = (flat_rows[2], flat_rows[0].shape[0], q.dtype, q.device)
            if batch is not last_batch or layout != last_layout:
                plan = plan_batch(batch, groups)
                last_batch, last_layout = batch, layout
            if plan is not None:
                if audit is not None:
                    audit["split_calls"] += 1
                return plan.apply(native, owner, q, keys, values, tables, flat_rows, result)
        return native(owner, q, keys, values, tables, groups, flat_rows=flat_rows, result=result)

    return forward


@contextmanager
def count_attention_calls(audit):
    """Count framework calls inside attention only, in the untimed audit burst."""
    attend = AscendEagerAttention.attend_metadata
    matmul, softmax = torch.matmul, torch.softmax

    def count_matmul(*args, **kwargs):
        audit["matmul"] += 1
        return matmul(*args, **kwargs)

    def count_softmax(scores, *args, **kwargs):
        batch, _, queries, keys = scores.shape
        audit["softmax_b_q_k"][f"{batch}x{queries}x{keys}"] += 1
        return softmax(scores, *args, **kwargs)

    def counted(*args, **kwargs):
        with (
            patch.object(torch, "matmul", count_matmul),
            patch.object(torch, "softmax", count_softmax),
        ):
            return attend(*args, **kwargs)

    with patch.object(AscendEagerAttention, "attend_metadata", counted):
        yield


@contextmanager
def route(enabled, audit, *, cold_count=3, plan_batch=ColdBatch.plan):
    if not enabled and audit is None:
        yield
        return
    execute = ModelRunner._execute

    def run(runner, schedule, prepared):
        phase = schedule.phase if schedule is not None else None
        if audit is not None:
            audit["steps"][phase.value if phase is not None else "mixed_or_unscheduled"] += 1
        if phase != RequestPhase.PREFILL:
            return execute(runner, schedule, prepared)
        spans = schedule.requests
        if audit is not None:
            audit["prefill_spans"].append(
                [[s.request.request_id, s.token_start, s.token_count] for s in spans]
            )
        mixed = (
            len(spans) == 4
            and sum(s.token_start == 0 for s in spans) == cold_count
            and all(
                s.token_start == s.request.prefill_window_origin
                and s.token_count > 1
                and s.token_start + s.token_count == len(s.request.prompt_token_ids)
                for s in spans
            )
        )
        with ExitStack() as hooks:
            if audit is not None:
                hooks.enter_context(count_attention_calls(audit))
            if enabled and mixed:
                hooks.enter_context(
                    patch.object(
                        AscendEagerAttention,
                        "_forward",
                        split_forward(AscendEagerAttention._forward, audit, plan_batch),
                    )
                )
            return execute(runner, schedule, prepared)

    with patch.object(ModelRunner, "_execute", run):
        yield


@contextmanager
def first_tokens(engine, targets, *, profiler, directory, mode):
    """Start after donor completion, stop when every target's first token is returned."""
    pending = {f"target-{i}" for i in range(len(targets))}
    step = engine.step
    with ExitStack() as capture:
        capture.enter_context(trace_target(profiler, directory))
        capture.enter_context(torch.profiler.record_function(f"RLT/{mode}/targets_to_first_tokens"))

        def observed_step():
            outputs = step()
            if pending:
                pending.difference_update(o.request_id for o in outputs if o.token_ids)
                if not pending:
                    capture.close()
            return outputs

        with patch.object(engine, "step", observed_step):
            yield
        if pending:
            raise RuntimeError(f"missing first tokens for {sorted(pending)}")


class ThreePlusOne(paired.Pairs):
    label = "THREE_PLUS_ONE"
    candidate = "three_plus_one"
    cold_count = 3
    batched_suffix_mode = None
    stock_methods = (*paired.Pairs.stock_methods, "_batched_prefill")
    audit_scope = "target prefill attention matmul calls and softmax [batch,queries,keys] shapes"
    route = staticmethod(route)

    @staticmethod
    def new_audit():
        return dict(
            steps=Counter(), prefill_spans=[], split_calls=0, matmul=0, softmax_b_q_k=Counter()
        )

    @classmethod
    def audit_issues(cls, audits, calls):
        issues = []
        for mode, arms in audits.items():
            for arm, audit in arms.items():
                split = arm == "reuse" and mode != "original"
                shapes = {"4x144x144": calls}
                if arm == "reuse":
                    hot = 4 - cls.cold_count
                    cold_batch = cls.cold_count if split else 1
                    hot_batch = hot if mode == cls.batched_suffix_mode else 1
                    shapes = {
                        f"{cold_batch}x144x144": calls * (cls.cold_count // cold_batch),
                        f"{hot_batch}x16x144": calls * (hot // hot_batch),
                    }
                spans = [
                    [f"target-{i}", 128, 16]
                    if i < 4 - cls.cold_count and arm == "reuse"
                    else [f"target-{i}", 0, 144]
                    for i in range(4)
                ]
                checks = {
                    "softmax_b_q_k": (dict(audit["softmax_b_q_k"]), shapes),
                    "matmul": (audit["matmul"], 2 * sum(shapes.values())),
                    "split_calls": (audit["split_calls"], calls if split else 0),
                    "prefill_spans": (audit["prefill_spans"], [spans]),
                    "observed_decode": (audit["steps"]["recurrent"] > 0, True),
                }
                issues.extend(
                    dict(mode=mode, arm=arm, check=k, actual=a, expected=e)
                    for k, (a, e) in checks.items()
                    if a != e
                )
        return issues

    def timing_summary(self):
        summary = super().timing_summary()
        per_request = {}
        for i, row in enumerate(self.runs[0]["original"]["reuse"]["requests"]):
            times = {
                mode: [p[mode]["reuse"]["requests"][i]["ttft_ms"] for p in self.runs[1:]]
                for mode in ("original", self.candidate)
            }
            per_request[f"target-{i}"] = dict(
                hit_tokens=row["hit_tokens"],
                **paired.paired_ms(times["original"], times[self.candidate], self.candidate),
            )
        summary["reuse_requests"] = per_request
        return summary

    def profile_pair(self, trial_args, trial_kwargs):
        profiler = npu_profiler()
        case = trial_args[2]
        traces = {}
        for mode in ("original", self.candidate):
            directory = self.trace_dir / mode
            directory.mkdir(parents=True)
            print(f"Collecting {mode} reuse trace in {directory}", flush=True)
            profile = partial(first_tokens, profiler=profiler, directory=directory, mode=mode)
            replay = self.run_variant(mode, None, trial_args, trial_kwargs, profile_target=profile)
            arms = {**self.runs[-1][mode], "reuse": replay}
            issues = list(_mismatches(case, [arms])) + paired.token_diffs(self.reference, replay)
            files = sorted(
                str(p.resolve()) for p in directory.rglob("trace_view.json") if p.stat().st_size
            )
            self.record("trace", mode=mode, arm="reuse", issues=issues, files=files)
            print(
                "THREE_PLUS_ONE_TRACE",
                json.dumps(dict(mode=mode, issues=issues, files=files)),
                flush=True,
            )
            if issues:
                raise paired.GateFailure(f"{mode} trace replay diverged: {issues}")
            if not files:
                raise RuntimeError(f"no nonempty trace_view.json under {directory}")
            traces[mode] = files
        return traces


if __name__ == "__main__":
    paired.main(pair_type=ThreePlusOne, description=__doc__)
