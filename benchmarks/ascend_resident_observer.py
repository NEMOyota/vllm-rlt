"""Host-side observations for the fixed-pool, local, fully reserved Ouro benchmark."""

import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import islice
from time import perf_counter
from unittest.mock import patch

from vllm_rlt.request import RequestPhase


def dump_lines(path, records):
    with path.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def page_counts(owners, raw_free, total):
    active = len(owners)
    exclusive = sum(n == 1 for n in owners.values())
    return dict(
        active=active,
        exclusive=exclusive,
        shared=active - exclusive,
        cache_only=total - active - raw_free,
        raw_free=raw_free,
    )


def audit_pages(cache, keys_by_id):
    """Check refs and sharing by physical page, depth, position and prefix digest."""
    owners = Counter()
    indexed = Counter()
    signatures = {}

    def check(block, signature):
        if not 0 <= block < cache.num_blocks:
            raise ValueError(f"KV block out of range: {block}")
        if signatures.setdefault(block, signature) != signature:
            raise ValueError(f"KV alias with incompatible logical prefix/depth/page: {block}")

    for rid in cache._allocations:
        seen = set()
        digests = keys_by_id[rid]
        for depth, table in enumerate(cache.plane_block_tables(rid)):
            for page, block in enumerate(table):
                if block in seen:
                    raise ValueError(f"duplicate page in allocation {rid}: {block}")
                seen.add(block)
                key = digests[page] if page < len(digests) else (rid, page)
                check(block, (depth, key))
                owners[block] += 1
    for key, group in cache._prefixes.items():
        if len(group) != cache.storage_depths:
            raise ValueError("incomplete prefix index depth tuple")
        for depth, block in enumerate(group):
            check(block, (depth, key))
            indexed[block] += 1
    free = set(cache._free_blocks)
    used = set(owners) | set(indexed)
    if (
        free & used
        or len(free) != len(cache._free_blocks)
        or free | used != set(range(cache.num_blocks))
    ):
        raise ValueError("KV physical partition mismatch")
    if any(ref != owners[b] + indexed[b] for b, ref in enumerate(cache._refs)):
        raise ValueError("KV reference count mismatch")
    counts = page_counts(owners, len(free), cache.num_blocks)
    if counts["cache_only"] != len(set(indexed) - set(owners)):
        raise ValueError("KV block conservation failed")
    return counts


def residency(events, wall, total):
    owners = Counter()
    active = {}
    levels = defaultdict(float)
    curve = []
    last = 0.0
    for time, action, rid, tables, raw_free, waiting in events:
        levels[len(active)] += time - last
        last = time
        if action == "admit":
            active[rid] = tables
            for table in tables:
                owners.update(table)
        elif action == "release":
            for table in active.pop(rid):
                for block in table:
                    owners[block] -= 1
                    if owners[block] == 0:
                        del owners[block]
        curve.append(
            dict(
                t=time,
                event=action,
                request=rid,
                resident=len(active),
                waiting=waiting,
                **page_counts(owners, raw_free, total),
            )
        )
    levels[len(active)] += wall - last
    peak = max((row["resident"] for row in curve), default=0)
    return curve, dict(
        resident_peak=peak,
        resident_mean=sum(k * v for k, v in levels.items()) / wall,
        residence_seconds=dict(levels),
        peak_active_blocks=max((r["active"] for r in curve), default=0),
        peak_shared_blocks=max((r["shared"] for r in curve), default=0),
        peak_exclusive_blocks=max((r["exclusive"] for r in curve), default=0),
        peak_cache_only_blocks=max((r["cache_only"] for r in curve), default=0),
        min_raw_free_blocks=min((r["raw_free"] for r in curve), default=total),
    )


def audit_attention(stack, strategy, counts):
    """Count actual host-side operator arguments; the audit worker owns this process."""
    from vllm_rlt.kernels import ascend_attention

    sdpa = ascend_attention._attend
    fia = strategy.fused_attention

    def observe_sdpa(query, key, value, *args, **kwargs):
        batch, _, q, _ = query.shape
        counts["sdpa", (q,) * batch, (key.shape[-2],) * batch] += 1
        return sdpa(query, key, value, *args, **kwargs)

    def observe_fia(query, key, value, **kwargs):
        ends = kwargs["actual_seq_lengths"]  # host lists supplied by AscendPagedPrefill
        lengths = tuple(end - start for start, end in zip([0, *ends[:-1]], ends, strict=True))
        counts["fia", lengths, tuple(kwargs["actual_seq_lengths_kv"])] += 1
        return fia(query, key, value, **kwargs)

    stack.enter_context(patch.object(ascend_attention, "_attend", observe_sdpa))
    stack.enter_context(patch.object(strategy, "fused_attention", observe_fia))


class Observer:
    """Observers are installed on this engine instance for the target window only."""

    def __init__(self, engine, prompts, audit_stream=None):
        self.engine = engine
        self.cache = engine.cache_manager
        self.audit_stream = audit_stream
        self.keys = {}
        if audit_stream:
            self.keys = {
                rid: list(self.cache._prefix_keys(prompt[:-1])) for rid, prompt in prompts.items()
            }
        self.requests = {rid: {} for rid in prompts}
        self.waiting = 0
        self.events = []
        self.steps = []
        self.counts = Counter(budget_deferrals=0, allocation_failures=0, evicted_entries=0)
        self.operator_shapes = Counter()
        self.step = None
        self.origin = None

    def now(self):
        return perf_counter() - self.origin

    def log(self, event, **fields):
        if self.audit_stream:
            self.audit_stream.write(json.dumps(dict(t=self.now(), event=event, **fields)) + "\n")

    def snapshot(self, event, **fields):
        if self.audit_stream:
            counts = audit_pages(self.cache, self.keys)
            self.log(event, **fields, blocks=counts)

    def transition(self, event, rid=None, tables=()):
        self.waiting += int(event == "submit") - int(event == "admit")
        waiting = self.waiting
        self.events.append((self.now(), event, rid, tables, len(self.cache._free_blocks), waiting))

    def install(self, stack: ExitStack):
        cache, scheduler, executor = self.cache, self.engine.scheduler, self.engine.executor
        commit, allocate, claim = scheduler._commit_admission, cache.allocate, cache._claim
        finish, submit = scheduler.finish, executor.submit

        def admit(request, plan):
            # The original call checks availability, pins hits and allocates private pages.
            before_fails = self.counts["allocation_failures"]
            ok = commit(request, plan)
            rid = request.request_id
            if ok:
                tables = cache.plane_block_tables(rid)
                self.requests[rid].update(admitted=self.now(), hits=plan.cached_tokens)
                self.transition("admit", rid, tables)
                self.snapshot(
                    "admit",
                    rid=rid,
                    tables=tables,
                    hits=plan.cached_tokens,
                    fresh=plan.required_blocks,
                    cached_claim=plan.cached_claim_blocks,
                    budget=plan.total_budget_blocks,
                )
            else:
                if self.counts["allocation_failures"] == before_fails:
                    self.counts["budget_deferrals"] += 1
                self.log(
                    "deferred",
                    rid=rid,
                    budget=plan.total_budget_blocks,
                    hits=plan.cached_tokens,
                    raw_free=len(cache._free_blocks),
                )
            return ok

        def alloc(*args, **kwargs):
            ok = allocate(*args, **kwargs)
            self.counts["allocation_failures"] += int(not ok)
            self.log("allocate", rid=args[0], success=ok)
            return ok

        def reclaim(count):
            old_size = len(cache._prefixes)
            capture = self.audit_stream and count > len(cache._free_blocks)
            old = list(cache._prefixes.items()) if capture else []
            result = claim(count)
            delta = old_size - len(cache._prefixes)
            self.counts["evicted_entries"] += delta
            if delta and self.audit_stream:
                self.log("evict", entries=[(k.hex(), blocks) for k, blocks in old[:delta]])
            return result

        def release(request, reason):
            rid = request.request_id
            result = finish(request, reason)
            self.transition("release", rid)
            self.snapshot("release", rid=rid)
            return result

        def dispatch(batch):
            stamp = self.now()
            items = batch.requests
            phase = batch.phase.value
            for item in items:
                self.requests[item.request.request_id].setdefault("first_execution", stamp)
            lengths = []
            if phase in ("prefill", "recurrent"):
                lengths = [i.token_start + i.token_count for i in items]
            rows = batch.num_tokens
            self.step = dict(
                t=stamp,
                phase=phase,
                batch_size=len(items),
                q_rows=rows,
                context_lengths=lengths,
                max_context=max(lengths, default=0),
                budget_utilization=rows / scheduler.config.max_num_batched_tokens,
                waiting=len(scheduler.queues[RequestPhase.WAITING]),
                residents=len(cache._allocations),
            )
            if phase == "prefill":
                self.step["spans"] = [
                    (i.request.request_id, i.token_start, i.token_count) for i in items
                ]
            self.steps.append(self.step)
            return submit(batch)

        if self.audit_stream:
            if cache.attention.packed_pool_prefill is not None and cache.backend == "ascend_eager":
                audit_attention(stack, cache.attention.packed_pool_prefill, self.operator_shapes)
            publish = cache.publish_prefix

            def publish_pages(rid, tokens, length, event=None):
                before = len(cache._prefixes)
                result = publish(rid, tokens, length, event)
                added = len(cache._prefixes) - before
                if added:
                    keys = list(islice(reversed(cache._prefixes), added))
                    self.snapshot(
                        "publish",
                        rid=rid,
                        entries=[(k.hex(), cache._prefixes[k]) for k in reversed(keys)],
                    )
                return result

            stack.enter_context(patch.object(cache, "publish_prefix", publish_pages))

        for obj, name, hook in (
            (scheduler, "_commit_admission", admit),
            (cache, "allocate", alloc),
            (cache, "_claim", reclaim),
            (scheduler, "finish", release),
            (executor, "submit", dispatch),
        ):
            stack.enter_context(patch.object(obj, name, hook))

    def counters(self):
        backend = self.cache.attention
        kernel = backend.kernel
        fields = (
            "sdpa_calls",
            "batched_prefill_calls",
            "ragged_calls",
            "fallback_calls",
            "continuation_calls",
        )
        return {
            **{f: getattr(kernel, f, 0) for f in fields},
            "fia_calls": getattr(backend.packed_pool_prefill, "calls", 0),
        }

    def collect(self, prompts, params):
        self.engine.model_runner.synchronize()
        self.origin = perf_counter()
        self.transition("start")
        self.snapshot(
            "start",
            index=[(k.hex(), v) for k, v in self.cache._prefixes.items()]
            if self.audit_stream
            else None,
        )
        for rid, tokens in prompts.items():
            self.requests[rid]["submitted"] = self.now()
            self.engine.add_request(rid, tokens, params)
            self.transition("submit", rid)
        before = self.counters()
        while self.engine.has_unfinished_requests():
            self.step = None
            if self.audit_stream:
                self.operator_shapes.clear()
            outputs = self.engine.step()
            visible = self.now()
            after = self.counters()
            if self.step is not None:
                self.step["paths"] = {k: after[k] - before[k] for k in before}
                if self.audit_stream:
                    self.step["operator_shapes"] = [
                        dict(
                            op=op,
                            batch_size=len(q),
                            q_rows=sum(q),
                            q_lengths=q,
                            kv_extents=kv,
                            calls=n,
                        )
                        for (op, q, kv), n in self.operator_shapes.items()
                    ]
                    if self.cache.backend == "ascend_eager":
                        for op, field in (("fia", "fia_calls"), ("sdpa", "sdpa_calls")):
                            actual = sum(
                                n for (name, _, _), n in self.operator_shapes.items() if name == op
                            )
                            if actual != self.step["paths"][field]:
                                raise ValueError(
                                    f"{op} audit counted {actual} calls; "
                                    f"backend reported {self.step['paths'][field]}"
                                )

                # RECURRENT is the attention-bearing decode phase in this Ouro configuration.
                direct = self.step["phase"] == "recurrent" and self.step["paths"]["ragged_calls"]
                if direct:
                    shape = self.step
                    shape["gather_kv_bytes_estimate_per_layer"] = (
                        2
                        * shape["q_rows"]
                        * shape["max_context"]
                        * self.cache.num_kv_heads
                        * self.cache.head_dim
                        * self.cache.key_cache.element_size()
                    )
            before = after
            for out in outputs:
                row = self.requests[out.request_id]
                if out.token_ids:
                    row.setdefault("first_token", visible)
                if out.finished:
                    row.update(
                        finished=visible,
                        tokens=out.token_ids,
                        exit_depths=out.exit_depths,
                        finish_reason=out.finish_reason,
                    )
        self.engine.model_runner.synchronize()
        wall = self.now()
        self.snapshot("drained")
        return wall
