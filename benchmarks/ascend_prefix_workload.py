"""One synchronous, single-device trial of an Ouro prefix-cache workload."""

import gc
import hashlib
import math
import random
import struct
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from benchmarks.ascend_prefix_cache import BLOCK_SIZE, generate
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.request import RequestPhase


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    prefix: int
    suffix: int
    concurrency: int = 1
    hits: int | None = None
    churn: int = 0  # Number of unrelated full-pool bursts between primer and targets.
    tight_pool: bool = False  # Match the resident control's pool to its eviction trial.

    def __post_init__(self):
        if self.prefix < 0 or self.suffix < 0 or self.prefix + self.suffix == 0:
            raise ValueError("prefix and suffix must form a nonempty prompt")
        if self.concurrency < 1 or self.churn < 0:
            raise ValueError("concurrency must be positive and churn nonnegative")
        if self.hits is not None and not 0 <= self.hits <= self.concurrency:
            raise ValueError("hits must be between zero and concurrency")

    @property
    def reusable_tokens(self):
        limit = min(self.prefix, self.prefix + self.suffix - 1)
        return limit // BLOCK_SIZE * BLOCK_SIZE


def _markers(rng, vocab, count, reserved):
    if count + len(reserved) > vocab:
        raise ValueError("vocabulary too small for distinct workload markers")
    candidates = rng.sample(range(vocab), count + len(reserved))
    return [token for token in candidates if token not in reserved][:count]


def _prompts(case, arm, output_tokens, vocab, seed):
    rng = random.Random(seed)
    count = case.concurrency
    hot = count if case.hits is None else case.hits
    # Unique marker IDs guarantee unrelated first pages and distinct first suffix
    # tokens independently of the random data. A miss changes only the primer.
    length = case.prefix + case.suffix
    # Preserve the legacy single-client target and primers for the same seed
    # and lengths. All arms submit identical targets and filler prompts.
    template = rng.choices(range(vocab), k=length)
    donor = template.copy()
    if case.suffix:
        donor[case.prefix] = (donor[case.prefix] + 1) % vocab
    miss_marker = (donor[0] + 1) % vocab
    reserved = {template[0], donor[0], miss_marker}
    if case.suffix:
        reserved.update((template[case.prefix], donor[case.prefix]))
    markers = _markers(rng, vocab, count, reserved)
    targets = []
    for index in range(count):
        prompt = template.copy()
        if index and case.suffix:
            prompt[case.prefix :] = rng.choices(range(vocab), k=case.suffix)
            prompt[case.prefix] = markers[index]
        if index >= hot:
            prompt[0] = markers[index]
        targets.append(prompt)
    if arm == "miss":
        donor[0] = miss_marker

    # With one output token these prompts reserve exactly as much KV as a target
    # generating its full output. Together they claim the entire tight pool.
    fillers = []
    for marker in _markers(rng, vocab, count * case.churn, reserved.union(markers)):
        prompt = rng.choices(range(vocab), k=length + output_tokens - 1)
        prompt[0] = marker
        fillers.append(prompt)
    return donor, targets, fillers


def _target_burst(engine, targets, params, expected):
    cache = engine.cache_manager
    hits_before, queries_before = cache.prefix_hits, cache.prefix_queries
    records = {f"target-{i}": {"expected_hit_tokens": hit} for i, hit in enumerate(expected)}
    submitted = {}
    for request_id, prompt in zip(records, targets, strict=True):
        submitted[request_id] = time.perf_counter()
        engine.add_request(request_id, prompt, params)

    block_tables = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        visible = time.perf_counter()
        batch = engine.last_schedule
        if batch.phase == RequestPhase.PREFILL:
            for span in batch.requests:
                record = records[span.request.request_id]
                if "prefill_start" not in record:
                    record["prefill_start"] = span.token_start
                    record["hit_tokens"] = span.request.prefill_window_origin
                    # All targets fit together with full output capacity reserved.
                    # Keep the immutable table and count unique blocks after timing.
                    block_tables.append(cache.plane_block_tables(span.request.request_id))
        for output in outputs:
            record = records[output.request_id]
            elapsed_ms = (visible - submitted[output.request_id]) * 1000
            if output.token_ids:
                record.setdefault("ttft_ms", elapsed_ms)
            if output.finished:
                if (
                    output.finish_reason != "length"
                    or len(output.token_ids) != params.max_tokens
                    or output.exit_depths != [4] * params.max_tokens
                ):
                    raise RuntimeError(f"{output.request_id} did not complete fixed-depth decoding")
                record.update(tokens=output.token_ids, e2e_ms=elapsed_ms)

    observed = cache.prefix_hits - hits_before
    queries = cache.prefix_queries - queries_before
    return {
        "requests": list(records.values()),
        "wall_ms": (visible - submitted["target-0"]) * 1000,
        "output_tokens": sum(len(record["tokens"]) for record in records.values()),
        "prefix_hits": observed,
        "prefix_queries": queries,
        "peak_used_blocks": len({b for tables in block_tables for plane in tables for b in plane}),
        "num_blocks": cache.num_blocks,
    }


def create_trial_engine(
    model, backend, case, arm, prompt_length, output_tokens, *, chunk_size=None
):
    """Build the matrix engine and fixed-depth greedy parameters for timing or replay."""
    capacity = prompt_length + output_tokens - 1
    blocks = model.config.total_ut_steps * math.ceil(capacity / BLOCK_SIZE)
    tight_pool = case.churn > 0 or case.tight_pool
    pool = blocks * (case.concurrency if tight_pool else case.concurrency + 1)
    gc.collect()  # Release cycles holding preceding trials' device buffers.
    engine = StageEngine(
        model,
        attention_backend=backend,
        cache_config=CacheConfig(
            num_blocks=pool, block_size=BLOCK_SIZE, enable_prefix_caching=arm != "off"
        ),
        execution_config=ExecutionConfig(multi_stream=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=case.concurrency,
            max_num_batched_tokens=case.concurrency * capacity,
            prefill_chunk_size=capacity if chunk_size is None else chunk_size,
            admission_scan_limit=max(case.concurrency, 64),
        ),
    )
    params = SamplingParams(
        max_tokens=output_tokens, temperature=0, min_loops=4, max_loops=4, ignore_eos=True
    )
    return engine, params


def prime_trial(engine, case, donor, targets, fillers, params):
    """Run the donor, then the tight-pool LRU lookup and churn bursts in matrix order."""
    prime_params = replace(params, max_tokens=1)
    generate(engine, "donor", donor, prime_params)
    primed_hit_tokens = None
    if case.churn > 0 or case.tight_pool:
        primed_hit_tokens = len(engine.cache_manager.lookup_prefix(targets[0])) * BLOCK_SIZE
    for start in range(0, len(fillers), case.concurrency):
        for index, prompt in enumerate(fillers[start : start + case.concurrency], start):
            engine.add_request(f"churn-{index}", prompt, prime_params)
        while engine.has_unfinished_requests():
            engine.step()
    return primed_hit_tokens


def expected_hits(case, arm):
    """Return the expected donor hit before churn and each target hit after churn."""
    hot = case.concurrency if case.hits is None else case.hits
    primed = case.reusable_tokens if arm == "reuse" and hot else 0
    return primed, [primed if i < hot and not case.churn else 0 for i in range(case.concurrency)]


@torch.inference_mode()
def run_trial(model, backend, case, arm, output_tokens, seed, *, target_context=None):
    """Prime a fresh engine, optionally evict the donor, then submit one target burst."""
    if arm not in ("off", "miss", "reuse") or output_tokens < 1:
        raise ValueError("expected off/miss/reuse arm and positive output length")
    donor, targets, fillers = _prompts(case, arm, output_tokens, model.config.vocab_size, seed)
    engine, params = create_trial_engine(model, backend, case, arm, len(donor), output_tokens)
    try:
        primed_hit_tokens = prime_trial(engine, case, donor, targets, fillers, params)
        expected_primed, expected = expected_hits(case, arm)
        with target_context() if target_context is not None else nullcontext():
            result = _target_burst(engine, targets, params, expected)
        for record, prompt in zip(result["requests"], targets, strict=True):
            packed = struct.pack(f"<{len(prompt)}q", *prompt)
            record["prompt_sha256"] = hashlib.sha256(packed).hexdigest()
        result.update(
            churn_requests=len(fillers),
            primed_hit_tokens=primed_hit_tokens,
            expected_primed_hit_tokens=expected_primed if primed_hit_tokens is not None else None,
        )
        return result
    finally:
        engine.close()
