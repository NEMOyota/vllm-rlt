"""Live greedy-margin experiment on synchronous prefix-cache matrix bursts."""

import math
import time
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_c4_slot import _digest
from benchmarks.ascend_prefix_workload import (
    _prompts,
    create_trial_engine,
    expected_hits,
    prime_trial,
)
from vllm_rlt.request import Operation, RequestPhase


def _top2_gap(logits):
    # Converting stored BF16 logits to FP32 is exact. Use FP32 TopK on Ascend
    # as well as CPU; only the final scalar crosses the device boundary.
    values = torch.topk(logits.float(), 2).values
    gap = (values[0] - values[1]).item()
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("nonfinite or unsorted top-two logits")
    return gap


@contextmanager
def _watch_samples(runner, sampled):
    execute = runner._execute_batch

    def watch(batch, prepared):
        if batch.operation != Operation.CODA:
            return execute(batch, prepared)
        rows = iter(zip(batch.spans, prepared.output_indices, strict=True))
        sample = runner.sampler.sample

        def guarded_sample(logits, params, generator):
            span, index = next(rows)
            if params.temperature != 0 or logits.ndim != 1 or logits.numel() < 2:
                raise ValueError("expected one greedy vocabulary row")
            # Only a real hit can trigger the cold replay. Transfer one scalar
            # for that row; ties in topk never determine the selected token ID.
            gap = _top2_gap(logits) if span.request.prefill_window_origin else None
            sampled.append((span.request.request_id, index + 1, gap))
            return sample(logits, params, generator)

        with patch.object(runner.sampler, "sample", guarded_sample):
            result = execute(batch, prepared)
        if next(rows, None) is not None:
            raise RuntimeError("target CODA omitted a real sampling row")
        return result

    with patch.object(runner, "_execute_batch", watch):
        yield


def _target_window(engine, targets, params, expected, threshold):
    cache = engine.cache_manager
    queries_before, hits_before = cache.prefix_queries, cache.prefix_hits
    ids = [f"target-{i}" for i in range(len(targets))]
    observed = {rid: [] for rid in ids}
    committed = observed.copy()
    hits = dict.fromkeys(ids)
    hashes = {rid: _digest(p) for rid, p in zip(ids, targets, strict=True)}
    first_ms = trigger = None
    sampled, trigger_step, schedule = [], [], []
    monitor = (
        _watch_samples(engine.model_runner, sampled) if threshold is not None else nullcontext()
    )
    with monitor:
        start = time.perf_counter()
        for rid, prompt in zip(ids, targets, strict=True):
            engine.add_request(rid, prompt, params)
        while engine.has_unfinished_requests():
            sampled.clear()
            outputs = engine.step()
            batch = engine.last_schedule
            spans = []
            for span in batch.requests:
                rid = span.request.request_id
                phase = batch.phase_for(span)
                spans.append([rid, phase.value, span.token_start, span.token_count, span.depth])
                if phase == RequestPhase.PREFILL and hits[rid] is None:
                    hits[rid] = span.request.prefill_window_origin
                    if span.token_start != hits[rid]:
                        raise RuntimeError(f"{rid}: first prefill offset differs from the hit")
            schedule.append([batch.loop_count, spans])
            for out in outputs:
                rid = out.request_id
                tokens = list(out.token_ids)
                if (
                    tokens[:-1] != observed[rid]
                    or out.exit_depths != [4] * len(tokens)
                    or out.finished != (len(tokens) == params.max_tokens)
                    or (out.finished and out.finish_reason != "length")
                ):
                    raise RuntimeError(f"{rid}: unexpected fixed-depth target output")
                observed[rid] = tokens
            if threshold is not None:
                if [(o.request_id, len(o.token_ids)) for o in outputs] != [
                    (rid, pos) for rid, pos, _ in sampled
                ]:
                    raise RuntimeError(
                        "sampled CODA rows differ from the synchronous visible outputs"
                    )
                for rid, pos, gap in sampled:
                    if gap is not None and gap <= threshold:
                        trigger = dict(
                            request_id=rid, position=pos, gap=gap, selected_id=observed[rid][-1]
                        )
                        break
                if trigger is not None:
                    trigger_step = [
                        dict(request_id=rid, position=pos, selected_id=observed[rid][-1])
                        for rid, pos, _ in sampled
                    ]
                    break  # Nothing from this entire engine step has been committed.
            if outputs:
                committed = observed.copy()  # Each next emission replaces its token list.
                if first_ms is None:
                    first_ms = (time.perf_counter() - start) * 1000
        elapsed = (time.perf_counter() - start) * 1000

    # Mixed batches may stop after admission and before every target's first
    # prefill. Count any live allocation and leave never-admitted requests null.
    for rid in ids:
        if hits[rid] is None and cache.has_allocation(rid):
            hits[rid] = engine.scheduler.requests[rid].prefill_window_origin
    allocated = [n for n in hits.values() if n is not None]
    if (
        cache.prefix_queries - queries_before != len(allocated)
        or cache.prefix_hits - hits_before != sum(allocated)
        or any(n is not None and n != want for n, want in zip(hits.values(), expected, strict=True))
    ):
        raise RuntimeError("unexpected target prefix hits or allocation counters")
    if trigger is None and any(len(tokens) != params.max_tokens for tokens in observed.values()):
        raise RuntimeError("native burst stopped before completing every target")
    return dict(
        tokens=observed,
        hit_tokens=hits,
        prompt_sha256=hashes,
        schedule=schedule,
        target_ms=elapsed,
        first_target_token_ms=first_ms,
        trigger=trigger,
        trigger_step=trigger_step,
        committed=committed,
    )


@torch.inference_mode()
def _burst(model, backend, case, contract, arm, threshold=None):
    output = contract["output_tokens"]
    donor, targets, fillers = _prompts(case, arm, output, model.config.vocab_size, contract["seed"])
    engine, params = create_trial_engine(model, backend, case, arm, len(donor), output)
    try:
        primed = prime_trial(engine, case, donor, targets, fillers, params)
        expected_primed, expected = expected_hits(case, arm)
        if primed is not None and primed != expected_primed:
            raise RuntimeError("unexpected donor hit before the target burst")
        return _target_window(engine, targets, params, expected, threshold)
    finally:
        engine.close()


def native_burst(model, backend, case, contract, arm):
    if arm not in ("off", "reuse"):
        raise ValueError("native guard control expects off or reuse")
    run = _burst(model, backend, case, contract, arm)
    return {
        k: run[k]
        for k in (
            "tokens",
            "hit_tokens",
            "prompt_sha256",
            "schedule",
            "target_ms",
            "first_target_token_ms",
        )
    }


def guarded_burst(model, backend, case, contract, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("guard threshold must be finite and nonnegative")
    hot = _burst(model, backend, case, contract, "reuse", threshold)
    cold = None
    conflicts = {}
    tokens = hot["tokens"]
    first_ms = hot["first_target_token_ms"]
    if hot["trigger"] is not None:
        cold = native_burst(model, backend, case, contract, "off")
        if list(cold["prompt_sha256"].items()) != list(hot["prompt_sha256"].items()):
            raise RuntimeError("cold fallback changed ordered target prompts")
        for rid, emitted in hot["committed"].items():
            for pos, (real, replayed) in enumerate(
                zip(emitted, cold["tokens"][rid][: len(emitted)], strict=True), 1
            ):
                if real != replayed:
                    conflicts[rid] = pos
                    break
        tokens = None if conflicts else cold["tokens"]
        # With no earlier commits the verified answer is available only after
        # this implementation completes the full cold burst and prefix check.
        if first_ms is None and not conflicts:
            first_ms = hot["target_ms"] + cold["target_ms"]
    cold_ms = cold["target_ms"] if cold is not None else 0.0
    return dict(
        threshold=threshold,
        trigger=hot["trigger"],
        trigger_step=hot["trigger_step"],
        committed_hot=hot["committed"],
        hot_observed=hot["tokens"],
        hot_schedule=hot["schedule"],
        hit_tokens=hot["hit_tokens"],
        prompt_sha256=hot["prompt_sha256"],
        cold=cold,
        tokens=tokens,
        prefix_conflicts=conflicts,
        fallback_used=cold is not None,
        hot_target_ms=hot["target_ms"],
        cold_target_ms=cold_ms,
        total_target_ms=hot["target_ms"] + cold_ms,
        first_committed_ms=first_ms,
    )
