"""Capture or replace one C4 request's valid logical KV between consecutive decode steps."""

import hashlib
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_page_control import _same_page
from vllm_rlt.request import RequestPhase


def _regions(after, shared, prompt):
    valid = prompt + after - 1
    if after == 1:
        return {"all": (0, valid), "prefix": (0, shared), "suffix": (shared, prompt)}
    return {"all": (0, valid), "prompt": (0, prompt), "generated": (prompt, valid)}


def _segments(table, start, end, block_size):
    for page in range(start // block_size, (end - 1) // block_size + 1):
        left = max(0, start - page * block_size)
        right = min(block_size, end - page * block_size)
        yield page, table[page], slice(left, right)


def _copy_to_host(parts):
    return tuple(t.detach().to(device="cpu", copy=True).contiguous() for t in parts)


def _update_hash(digest, pair):
    for value in pair:
        digest.update(value.contiguous().view(torch.uint8).numpy())


class FocusHistoryKV:
    """Observe CODA output #1/#7; only write into a cold target with private pages."""

    def __init__(
        self,
        engine,
        ids,
        focus,
        prompt_tokens,
        shared_tokens,
        *,
        capture=False,
        after=1,
        region="all",
        source=None,
    ):
        self.engine = engine
        self.cache = engine.cache_manager
        self.ids = tuple(ids)
        if len(set(self.ids)) != 4 or len(self.ids) != 4 or focus not in self.ids:
            raise ValueError("expected four distinct targets including the focus")
        if self.cache.layout != "last_exited" or self.cache.dtype != torch.bfloat16:
            raise ValueError("requires a BF16 last_exited cache")
        if not 0 < shared_tokens < prompt_tokens or after not in (1, 7):
            raise ValueError("expected a nonempty prefix/suffix and cut after token 1 or 7")
        if region not in _regions(after, shared_tokens, prompt_tokens):
            raise ValueError("region is unavailable at this cut")
        if capture and source is not None:
            raise ValueError("a capture cannot accept replacement KV")
        if not capture and self.cache.enable_prefix_caching:
            raise ValueError("page writes require a cold private destination")
        self.focus = focus
        self.prompt = prompt_tokens
        self.shared = shared_tokens
        self.capture = capture
        self.after = after
        self.region = region
        self.source = source
        self.snapshots = {}
        self.histories = {}
        self.boundaries = {}
        self._emitted = 0
        self._hook = None
        self._finished = False

    def _tables(self, valid):
        cache = self.cache
        if set(cache._allocations) != set(self.ids):
            raise RuntimeError("cut requires all four targets to be the only live requests")
        tables = cache.plane_block_tables(self.focus)
        count = (valid + cache.block_size - 1) // cache.block_size
        if len(tables) != cache.max_loops or any(len(t) < count for t in tables):
            raise RuntimeError("missing focus KV planes or pages")
        blocks = [block for plane in tables for block in plane]
        if len(set(blocks)) != len(blocks) or any(not 0 <= b < cache.num_blocks for b in blocks):
            raise RuntimeError("invalid or duplicate focus physical blocks")
        if not self.capture:
            private = set(blocks)
            for rid in self.ids:
                if rid != self.focus and any(
                    b in private for plane in cache.plane_block_tables(rid) for b in plane
                ):
                    raise RuntimeError("destination focus allocation aliases another request")
        # The known C4 run writes every logical position in order, in every
        # depth and layer. The just-emitted token has not yet entered the KV.
        allocation = cache._get_allocation(self.focus)
        for depth, plane in enumerate(allocation.written):
            if any(w.prefix != valid or w.pending for w in plane):
                raise RuntimeError(f"unexpected focus KV extent at depth {depth}")
        return tuple(tuple(plane[:count]) for plane in tables)

    def _snapshot(self, tables, valid):
        pools = (self.cache.key_cache, self.cache.value_cache)
        return tuple(
            tuple(
                _copy_to_host(pool[block, :, span] for pool in pools)
                for _, block, span in _segments(table, 0, valid, self.cache.block_size)
            )
            for table in tables
        )

    def _validate_source(self, source, tables, valid):
        cache = self.cache
        if len(source) != len(tables):
            raise ValueError("source and destination have different depth counts")
        for plane, table in zip(source, tables, strict=True):
            if len(plane) != len(table):
                raise ValueError("source and destination have different logical page counts")
            for i, pair in enumerate(plane):
                tokens = min(cache.block_size, valid - i * cache.block_size)
                shape = (cache.num_layers, tokens, cache.num_kv_heads, cache.head_dim)
                if len(pair) != 2 or any(
                    t.shape != shape or t.device.type != "cpu" or t.dtype != cache.dtype
                    for t in pair
                ):
                    raise ValueError("expected a source of cropped BF16 host KV pages")

    def _visit(self, tables, source, start, end):
        before, selected, after = (hashlib.sha256() for _ in range(3))
        changed = checked = 0
        pools = (self.cache.key_cache, self.cache.value_cache)
        for depth, table in enumerate(tables):
            for page, block, span in _segments(table, start, end, self.cache.block_size):
                checked += 1
                source_page = (
                    None if source is None else tuple(t[:, span] for t in source[depth][page])
                )
                if self.capture:
                    # Capture mode owns a full host snapshot and computes all region
                    # digests from that one read, including the cropped final page.
                    _update_hash(before, source_page)
                    continue
                device = tuple(pool[block, :, span] for pool in pools)
                original = _copy_to_host(device)
                replacement = original if source_page is None else source_page
                changed += int(not _same_page(original, replacement))
                _update_hash(before, original)
                _update_hash(selected, replacement)
                for dst, src in zip(device, replacement, strict=True):
                    dst.copy_(src.contiguous())
                verified = _copy_to_host(device)
                if not _same_page(verified, replacement):
                    raise RuntimeError(f"readback failed at depth {depth}, logical page {page}")
                _update_hash(after, verified)
        original = before.hexdigest()
        return dict(
            checked_pages=checked,
            changed_pages=changed,
            start=start,
            end=end,
            before_sha256=original,
            source_sha256=original if self.capture else selected.hexdigest(),
            after_sha256=original if self.capture else after.hexdigest(),
        )

    @torch.no_grad()
    def _observe(self, outputs):
        if not outputs or self._emitted >= 7:
            return
        self._emitted += 1
        if (
            self.engine.last_schedule.phase != RequestPhase.CODA
            or tuple(o.request_id for o in outputs) != self.ids
            or any(len(o.token_ids) != self._emitted for o in outputs)
        ):
            raise RuntimeError("expected one CODA output per target in the original C4 order")
        if self._emitted not in (1, 7):
            return
        if any(o.finished or len(o.prompt_token_ids) != self.prompt for o in outputs):
            raise RuntimeError("expected live C4 targets with matching prompt lengths")
        key = str(self._emitted)
        self.histories[key] = {o.request_id: list(o.token_ids) for o in outputs}
        if not self.capture and self._emitted != self.after:
            return
        valid = self.prompt + self._emitted - 1
        tables = self._tables(valid)
        regions = _regions(self._emitted, self.shared, self.prompt)
        source = self.source
        if self.capture:
            source = self.snapshots[self._emitted] = self._snapshot(tables, valid)
        else:
            regions = {self.region: regions[self.region]}
            if source is not None:
                self._validate_source(source, tables, valid)
        self.boundaries[key] = {
            name: self._visit(tables, source, start, end) for name, (start, end) in regions.items()
        }

    def __enter__(self):
        if self._hook is not None or self.engine is None:
            raise RuntimeError("use a fresh focus history control for each replay")
        step = self.engine.step

        def observe_step():
            outputs = step()
            self._observe(outputs)
            return outputs

        self._hook = patch.object(self.engine, "step", observe_step)
        self._hook.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._hook.stop()
        self._hook = self.engine = self.cache = self.source = None
        if exc_type is None:
            if set(self.histories) != {"1", "7"}:
                raise RuntimeError("incomplete CODA history coverage at output 1 and 7")
            self._finished = True

    def report(self):
        if not self._finished:
            raise RuntimeError("finish the C4 replay before reading this report")
        return dict(histories=self.histories, boundaries=self.boundaries)
