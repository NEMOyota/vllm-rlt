"""Cold C4 slot and future-token controls on the real synchronous Ouro engine."""

import hashlib
import struct
from contextlib import ExitStack
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe, _snapshot
from benchmarks.ascend_prefix_replay import _prime_and_record, _target_steps
from benchmarks.ascend_prefix_workload import _prompts, create_trial_engine
from vllm_rlt.request import RequestPhase


def _digest(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}q", *tokens)).hexdigest()


@torch.inference_mode()
def cold_slot_run(model, backend, case, contract, variant, *, observe_attention=False):
    """Keep cold batch order/shape and donor fixed; edit one future token or clone a full target."""
    if variant not in ("base", "repeat", "future", "clones"):
        raise ValueError(f"unknown C4 slot variant: {variant}")
    p, n = case.reusable_tokens, case.prefix + case.suffix
    if (
        case.concurrency != 4
        or case.hits not in (None, 4)
        or case.churn
        or p != case.prefix
        or not 0 < p < n
    ):
        raise ValueError(
            "expected four shared-prefix targets with a page-aligned prefix and suffix"
        )
    if observe_attention and backend != "ascend_eager":
        raise ValueError("frozen attention requires an actual ascend_eager backend")

    output_tokens = contract["output_tokens"]
    vocab = model.config.vocab_size
    donor, original, fillers = _prompts(case, "off", output_tokens, vocab, contract["seed"])
    targets = [row.copy() for row in original]
    ids = [f"target-{i}" for i in range(len(targets))]
    if any(len(row) != n or row[:p] != original[2][:p] for row in targets):
        raise ValueError("C4 targets must share the same prefix and total length")

    edit = None
    if variant == "future":
        old = targets[2][p]
        new = (old + 1) % vocab
        if old == new:
            raise ValueError("failed to generate a distinct future token")
        targets[2][p] = new
        edit = dict(request_id=ids[2], position_0based=p, old_token=old, new_token=new)
    elif variant == "clones":
        targets = [original[2].copy() for _ in ids]

    engine, params = create_trial_engine(model, backend, case, "off", n, output_tokens)
    try:
        donor_spans = _prime_and_record(engine, case, donor, targets, fillers, params, None)
        cache = engine.cache_manager
        probe = LogicalKVProbe(cache, p, ids)
        prefill = {}
        step = engine.step

        def observe_step():
            outputs = step()
            batch = engine.last_schedule
            if batch.phase != RequestPhase.PREFILL:
                if not prefill:
                    raise RuntimeError("target work preceded the expected full C4 prefill")
                return outputs
            spans = [(s.request.request_id, s.token_start, s.token_count) for s in batch.requests]
            if prefill or batch.loop_count != 4 or spans != [(rid, 0, n) for rid in ids]:
                raise RuntimeError(
                    f"unexpected C4 cold prefill geometry: loops={batch.loop_count}, {spans}"
                )
            # All four recurrent depths have written their real prefix KV in this
            # synchronous step. No next step has consumed a generated token.
            for rid in ids:
                prefill[rid] = dict(_snapshot(cache, rid, p), tap="after_prefill")
            return outputs

        with ExitStack() as context:
            context.enter_context(probe)
            context.enter_context(patch.object(engine, "step", observe_step))
            if observe_attention:
                from benchmarks.ascend_prefix_attention_replay import FrozenAttention

                attention = context.enter_context(FrozenAttention(cache, ids, p, n))
            result = _target_steps(engine, targets, params, [0] * len(ids))
        if set(prefill) != set(ids):
            raise RuntimeError("missing C4 prefill endpoint")
        fingerprints = probe.report()
        result.update(
            case=case.name,
            variant=variant,
            mode="off",
            output_tokens_per_request=output_tokens,
            donor_prefill_spans=donor_spans,
            donor_sha256=_digest(donor),
            prefill_chunk_size=engine.scheduler.config.prefill_chunk_size,
            max_num_batched_tokens=engine.scheduler.config.max_num_batched_tokens,
            num_blocks=cache.num_blocks,
            future_edit=edit,
            prefix_sha256={rid: _digest(row[:p]) for rid, row in zip(ids, targets, strict=True)},
            prefill_fingerprints=prefill,
            fingerprints={rid: fingerprints[rid] for rid in ids},
        )
        if observe_attention:
            result["attention"] = attention.report()
        return result
    finally:
        engine.close()
