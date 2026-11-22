"""Correctness probes for the mainline Ascend packed host-metadata path."""

from collections import Counter
from contextlib import contextmanager
from itertools import groupby
from unittest.mock import patch

import torch
import torch.nn.functional as F

from vllm_rlt.core.kv_cache_manager import _WrittenPositions


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@torch.inference_mode()
def page_oracle(cache):
    """Two layers, four depths, one shared page and three different continuations."""
    block = cache.block_size
    require(block == 16 and cache.num_layers == 2 and cache.storage_depths == 4, "oracle geometry")
    prompt = list(range(18))
    require(cache.allocate("donor", len(prompt)), "donor allocation")

    def values(items):
        tensor = torch.tensor(items, device=cache.device, dtype=cache.dtype)
        return tensor[:, None, None].expand(-1, cache.num_kv_heads, cache.head_dim).contiguous()

    def write_donor(depth, layer, pos):
        batch = cache._prepare_batch(["donor"] * len(pos), [depth] * len(pos), list(pos))
        v = values([8 * 2 ** (depth + 4 * layer)] * len(pos))
        cache._write_prepared(layer, batch, torch.zeros_like(v), v)

    for depth in range(4):
        for layer in range(2):
            # Leave position zero missing from just one layer in the last plane.
            start = int((depth, layer) == (3, 1))
            write_donor(depth, layer, range(start, block))
    cache.publish_prefix("donor", prompt, block)
    require(not cache.lookup_prefix(prompt), "published a page with a hole in the last layer")
    write_donor(3, 1, [0])
    cache.publish_prefix("donor", prompt, block)
    prefix = cache.lookup_prefix(prompt)
    require(len(prefix) == 1 and len(set(prefix[0])) == 4, "expected one full four-depth page")

    for rid in ("a", "b"):
        require(cache.allocate(rid, 18, prefix=prefix), "shared prefix allocation")
    require(cache.allocate("cold", 2), "cold allocation")
    cache.free("donor")
    pages = torch.tensor(prefix[0], device=cache.device)
    shared_keys = cache.key_cache.index_select(0, pages)
    shared_values = cache.value_cache.index_select(0, pages)
    ids = ["a", "a", "b", "b", "cold", "cold"]
    positions = [16, 17, 16, 17, 0, 1]
    for depth in range(4):
        a, b, cold = (cache.get_block_table(rid, depth) for rid in ("a", "b", "cold"))
        require(a[0] == b[0] == prefix[0][depth], "incorrect shared KV address")
        require(len({a[0], a[1], b[1], cold[0]}) == 4, "private suffix pages alias")
        batch = cache._prepare_batch(ids, [depth] * 6, positions, packed_prefill=True)
        require(batch.cu_seqlens_q.tolist() == [0, 2, 4, 6], "incorrect packed query bounds")
        for layer in range(2):
            # With zero Q and K each answer is the mean of visible V rows.
            # a and b share sixteen 8s, but have different private suffixes.
            scale = 2 ** (depth + 4 * layer)
            v = values([25, 27, 59, 65, 2, 6]) * scale
            zeros = torch.zeros_like(v)
            cache._write_prepared(layer, batch, zeros, v)
            actual = cache._attend_prepared(layer, batch, zeros)
            expected = values([9, 10, 11, 14, 2, 4]) * scale
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    require(torch.equal(cache.key_cache.index_select(0, pages), shared_keys), "shared K changed")
    require(
        torch.equal(cache.value_cache.index_select(0, pages), shared_values), "shared V changed"
    )
    for rid in ("a", "b", "cold"):
        cache.free(rid)
    require(cache.num_used_blocks == 0, "finished requests retained live KV")


class PrefillAudit:
    """Observe only the target prefill step; keep no Q/K/V tensors in the report."""

    def __init__(self, cache, starts, length):
        self.cache = cache
        self.starts = starts
        self.length = length
        self.depths = Counter()
        self.range_calls = Counter()
        self.sdpa_shapes = Counter()
        self.sdpa_rows = 0
        self.descriptors = set()
        self.checked_masks = set()

    def check_descriptor(self, batch):
        require(batch.cu_seqlens_q is not None, "target prefill used per-token page tables")
        names = {id(alloc): rid for rid, alloc in batch.allocations}
        tables = batch.block_tables.tolist()
        require(len(tables) == len(self.starts), "expected one page-table row per sequence")
        groups = groupby(batch.rows, key=lambda row: (id(row[0]), row[1]))
        ends = [0]
        observed = []
        for table_row, ((alloc_id, depth), rows) in enumerate(groups):
            rid = names[alloc_id]
            require(rid in self.starts and rid not in observed, f"unexpected packed request {rid}")
            positions = [position for _, _, position in rows]
            start = self.starts[rid]
            require(
                positions == list(range(start, self.length)),
                f"{rid} expected query range [{start},{self.length}), "
                f"got {len(positions)} rows from {positions[0]} to {positions[-1]}",
            )
            ends.append(ends[-1] + len(positions))
            full_table = self.cache.get_block_table(rid, depth)
            used = (self.length + self.cache.block_size - 1) // self.cache.block_size
            require(tables[table_row][:used] == list(full_table[:used]), f"{rid} wrong packed KV")
            observed.append(rid)
            self.depths[rid, depth] += 1
        require(sorted(observed) == sorted(self.starts), "wrong requests in packed prefill")
        require(batch.cu_seqlens_q.tolist() == ends, "wrong packed query offsets")
        require(batch.context_lengths.tolist() == [self.length] * len(observed), "wrong KV lengths")

    def check_mask(self, width, length, mask, causal):
        require(length == self.length, "incorrect SDPA context length")
        if mask is None:
            require(
                (causal and width == length) or (not causal and width == 1),
                f"missing absolute-position mask for Q={width}, K={length}",
            )
            return
        require(not causal and mask.dtype == torch.bool, "expected an explicit boolean SDPA mask")
        if mask not in self.checked_masks:
            positions = torch.arange(length - width, length)
            expected = torch.arange(length)[None, :] <= positions[:, None]
            require(
                mask.shape[-2:] == expected.shape and bool((mask.cpu() == expected).all()),
                f"incorrect absolute-position mask for Q={width}, K={length}",
            )
            self.checked_masks.add(mask)

    @contextmanager
    def observe(self):
        kernel = self.cache.attention.kernel
        attend = kernel.attend_metadata
        sdpa = F.scaled_dot_product_attention
        add_range = _WrittenPositions.add_range

        def observe_attention(q, k, v, batch):
            if batch not in self.descriptors:
                self.check_descriptor(batch)
                self.descriptors.add(batch)
            return attend(q, k, v, batch)

        def observe_sdpa(q, k, v, *args, **kwargs):
            require(
                all(t.dtype == torch.bfloat16 and t.device == self.cache.device for t in (q, k, v)),
                "unexpected SDPA device or dtype",
            )
            self.check_mask(
                q.shape[-2],
                k.shape[-2],
                kwargs.get("attn_mask", args[0] if args else None),
                kwargs.get("is_causal", False),
            )
            self.sdpa_shapes[f"{q.shape[0]}x{q.shape[-2]}x{k.shape[-2]}"] += 1
            self.sdpa_rows += q.shape[0] * q.shape[-2]
            return sdpa(q, k, v, *args, **kwargs)

        def observe_range(written, start, end):
            add_range(written, start, end)
            self.range_calls[id(written), start, end] += 1

        with (
            patch.object(kernel, "attend_metadata", observe_attention),
            patch.object(F, "scaled_dot_product_attention", observe_sdpa),
            patch.object(_WrittenPositions, "add_range", observe_range),
        ):
            yield

    def result(self):
        expected_ranges = Counter()
        for rid, start in self.starts.items():
            for depth, plane in enumerate(self.cache._get_allocation(rid).written):
                require(self.depths[rid, depth] == 1, f"{rid} missing or duplicate depth {depth}")
                for layer, written in enumerate(plane):
                    expected_ranges[id(written), start, self.length] += 1
                    require(
                        written.prefix == self.length, f"{rid} depth {depth} layer {layer} coverage"
                    )
        require(self.range_calls == expected_ranges, "per-request/depth/layer write range mismatch")
        expected_rows = sum(self.length - start for start in self.starts.values())
        require(
            self.sdpa_rows == expected_rows * self.cache.storage_depths * self.cache.num_layers,
            "SDPA did not cover every target prefill query at every layer and depth",
        )
        return {
            "packed_descriptors": len(self.descriptors),
            "sdpa_BxQxK": dict(self.sdpa_shapes),
            "explicit_masks_checked": len(self.checked_masks),
            "range_updates": sum(self.range_calls.values()),
        }
