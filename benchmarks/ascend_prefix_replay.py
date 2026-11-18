"""Replay one matrix burst with its original target order and capture its schedule."""

import hashlib
import struct
from contextlib import nullcontext

import torch

from benchmarks.ascend_prefix_workload import (
    Case,
    _prompts,
    create_trial_engine,
    expected_hits,
    prime_trial,
)
from vllm_rlt.request import RequestPhase


def _target_steps(engine, targets, params, expected):
    cache = engine.cache_manager
    hits_before, queries_before = cache.prefix_hits, cache.prefix_queries
    records = {}
    for i, prompt in enumerate(targets):
        rid = f"target-{i}"
        packed = struct.pack(f"<{len(prompt)}q", *prompt)
        records[rid] = dict(
            request_id=rid,
            prompt_sha256=hashlib.sha256(packed).hexdigest(),
            expected_hit_tokens=expected[i],
            first_prefill=None,
            tokens=[],
        )
        engine.add_request(rid, prompt, params)

    steps = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        batch = engine.last_schedule
        step = len(steps) + 1
        spans = []
        for item in batch.requests:
            rid = item.request.request_id
            phase = batch.phase_for(item)
            span = dict(
                request_id=rid,
                phase=phase.value,
                token_start=item.token_start,
                token_count=item.token_count,
                depth=item.depth,
            )
            spans.append(span)
            row = records[rid]
            if phase == RequestPhase.PREFILL and row["first_prefill"] is None:
                row["first_prefill"] = dict(span, step=step, loop_count=batch.loop_count)
                row["prefill_start"] = item.token_start
                row["hit_tokens"] = item.request.prefill_window_origin

        visible = []
        for out in outputs:
            row = records[out.request_id]
            # RequestOutput carries the cumulative sequence, including in a final event.
            new_tokens = out.token_ids[len(row["tokens"]) :]
            visible.append(
                dict(request_id=out.request_id, token_ids=new_tokens, finished=out.finished)
            )
            row["tokens"] = list(out.token_ids)
            if out.finished:
                row.update(exit_depths=list(out.exit_depths), finish_reason=out.finish_reason)
        steps.append(
            dict(
                step=step,
                phase=batch.phase.value if batch.phase is not None else "mixed",
                loop_count=batch.loop_count,
                spans=spans,
                visible=visible,
            )
        )
    return {
        "requests": list(records.values()),
        "prefix_hits": cache.prefix_hits - hits_before,
        "prefix_queries": cache.prefix_queries - queries_before,
        "steps": steps,
    }


@torch.inference_mode()
def replay_mode(
    model,
    backend,
    case: Case,
    mode,
    output_tokens,
    seed,
    *,
    target_context=None,
    engine_context=None,
):
    """Replay off/miss/reuse, or an uncached target split at the matching full-page boundary.

    target_context(engine, targets, donor) can observe the target burst. targets is ordered
    like the requests in the matrix report. The primer is complete before this context starts.
    engine_context(engine, targets, donor) covers the donor, fillers and targets for KV checks.
    Both contexts unwind before the engine closes, including on failure.
    """
    if mode not in ("off", "miss", "reuse", "split") or output_tokens < 1:
        raise ValueError("expected off, miss, reuse or split mode and positive output_tokens")
    arm = "off" if mode == "split" else mode
    donor, targets, fillers = _prompts(case, arm, output_tokens, model.config.vocab_size, seed)
    chunk_size = case.reusable_tokens if mode == "split" else None
    if chunk_size == 0:
        raise ValueError("split replay requires at least one full reusable prefix page")
    engine, params = create_trial_engine(
        model, backend, case, arm, len(donor), output_tokens, chunk_size=chunk_size
    )
    try:
        lifetime = engine_context(engine, targets, donor) if engine_context else nullcontext()
        with lifetime:
            prime_trial(engine, case, donor, targets, fillers, params)
            _, expected = expected_hits(case, arm)
            context = target_context(engine, targets, donor) if target_context else nullcontext()
            with context:
                result = _target_steps(engine, targets, params, expected)

        return dict(
            result,
            case=case.name,
            mode=mode,
            prefill_chunk_size=engine.scheduler.config.prefill_chunk_size,
            max_num_batched_tokens=engine.scheduler.config.max_num_batched_tokens,
            num_blocks=engine.cache_manager.num_blocks,
        )
    finally:
        engine.close()
