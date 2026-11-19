"""Compare native cold attention with a read-only continuation on the same Q/K/V."""

from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_layer0 import _compare
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention


class FrozenAttention:
    """Observe a complete equal-length C4 prefill without changing its returned outputs.

    The real cold kernel gathers current rows from write slots. The separate shadow
    kernel gathers the same logical history via block tables. For each immutable
    plan we check actual cold slots, actual hot group row order/positions and staged
    block IDs against each request's host page table. Both real masks are compared
    with the absolute causal mask. These are address and mask checks, not a second
    evaluator of the matmuls or a comparison of KV across different requests.
    """

    def __init__(self, cache, request_ids, prefix, total):
        self.ids = tuple(request_ids)
        if len(self.ids) != 4 or len(set(self.ids)) != 4 or not 0 < prefix < total - 1:
            raise ValueError("expected four distinct requests with a multi-row suffix")
        if cache.layout != "last_exited" or cache.dtype != torch.bfloat16:
            raise ValueError("expected BF16 last_exited KV")
        self.cache = cache
        self.native = cache.attention.kernel
        if not isinstance(self.native, AscendEagerAttention):
            raise ValueError("frozen attention requires the actual Ascend eager kernel")
        self.prefix = prefix
        self.total = total
        self.suffix = total - prefix
        self.expected = cache.max_loops * cache.num_layers
        self.checked = 0
        self._seen = set()
        self._issues = []
        self._first_gather = None
        self._first_output = None
        self._requests = {
            rid: dict(first_diff=None, different_calls=0, different_elements=0, max_abs=0.0)
            for rid in self.ids
        }
        self._cold = self._hot = self._patch = None
        self._done = False
        self.shadow = AscendEagerAttention()

    def _issue(self, name):
        if name not in self._issues:
            self._issues.append(name)

    def _mismatch(self, kind, depth, layer, rid=None, **details):
        self._issue("gather_or_mask_mismatch")
        if self._first_gather is None:
            self._first_gather = dict(
                kind=kind, depth=depth, layer=layer, request_id=rid, **details
            )

    def _prepare_hot(self, batch, depth):
        rows = batch.rows
        allocations = dict(batch.allocations)
        for i, rid in enumerate(self.ids):
            group = rows[i * self.total : (i + 1) * self.total]
            if rid not in allocations or any(
                alloc is not allocations[rid] or d != depth or pos != j
                for j, (alloc, d, pos) in enumerate(group)
            ):
                self._issue("unexpected_cold_row_order")
                return False
        ids = [rid for rid in self.ids for _ in range(self.suffix)]
        positions = list(range(self.prefix, self.total)) * len(self.ids)
        self._hot = self.cache._prepare_batch(ids, [depth] * len(ids), positions, for_write=False)
        self._cold = batch
        return True

    def _check_plan(self, batch, depth, layer):
        # The flattened view starts at the current layer's own storage offset.
        pool = self.cache.key_cache[:, layer]
        page_stride = pool.stride(0) // pool.stride(1)
        block_size = self.cache.block_size
        positions = torch.arange(self.total)
        future = positions[None, :] > positions[:, None]
        causal = torch.zeros(future.shape, dtype=torch.bfloat16)
        causal.masked_fill_(future, torch.finfo(causal.dtype).min)
        if not torch.equal(self.native._prefill_mask.cpu(), causal):
            self._mismatch("cold_mask", depth, layer)

        cold_blocks = batch.write_blocks.cpu().tolist()
        cold_offsets = batch.write_offsets.cpu().tolist()
        cold_slots = self.native._write_slots.cpu().tolist()
        tables = self._hot.block_tables.cpu()
        if len(self.shadow._groups) != len(self.ids):
            self._mismatch("shadow_group_count", depth, layer)
            return
        for i, (rid, group) in enumerate(zip(self.ids, self.shadow._groups, strict=True)):
            first, indices, history, mask, initial, lengths = group
            start = i * self.suffix
            if (
                initial
                or first != start
                or indices.tolist() != list(range(start, start + self.suffix))
                or history.tolist() != list(range(self.total))
                or lengths.tolist() != list(range(self.prefix + 1, self.total + 1))
            ):
                self._mismatch("shadow_group_positions", depth, layer, rid)
                continue
            if not torch.equal(mask.cpu(), causal[self.prefix :]):
                self._mismatch("shadow_mask", depth, layer, rid)
            logical = self.cache.plane_block_tables(rid)[depth]
            hot_table = tables[first].tolist()
            for pos in range(self.total):
                page, offset = divmod(pos, block_size)
                block = logical[page]
                slot = block * page_stride + offset
                row = i * self.total + pos
                if not 0 <= block < self.cache.num_blocks:
                    self._mismatch("logical_block_out_of_range", depth, layer, rid, position=pos)
                    break
                if (cold_blocks[row], cold_offsets[row], cold_slots[row]) != (block, offset, slot):
                    self._mismatch("cold_slot", depth, layer, rid, position=pos)
                    break
                # _forward indexes tables[first, history//block_size], then
                # block*page_stride + history%block_size. Validate before clamping.
                if hot_table[page] != block:
                    self._mismatch("shadow_block", depth, layer, rid, position=pos)
                    break

    def _compare_outputs(self, depth, layer, cold, hot):
        cold = cold.detach().cpu()
        hot = hot.detach().cpu()
        for rid, left, right in zip(self.ids, cold, hot, strict=True):
            stats = _compare(left.contiguous(), right.contiguous())
            entry = self._requests[rid]
            if stats["nonfinite_pairs"]:
                self._issue("nonfinite_attention_output")
            if stats["max_abs"] is not None:
                entry["max_abs"] = max(entry["max_abs"], stats["max_abs"])
            count = stats["different_elements"]
            if count:
                entry["different_elements"] += count
                entry["different_calls"] += 1
                detail = dict(depth=depth, layer=layer, request_id=rid, **stats)
                if entry["first_diff"] is None:
                    entry["first_diff"] = detail
                if self._first_output is None:
                    self._first_output = detail

    def __enter__(self):
        if self._patch is not None or self._done:
            raise RuntimeError("use a fresh frozen attention probe for each target burst")
        attend = self.cache._attend_prepared

        def observe(layer, batch, q):
            before = self.native.batched_prefill_calls
            output = attend(layer, batch, q)
            if len(getattr(batch, "rows", ())) != len(self.ids) * self.total:
                return output
            depth = batch.rows[0][1]
            new_plan = self._cold is not batch
            if new_plan and not self._prepare_hot(batch, depth):
                return output
            if (
                self.native.batched_prefill_calls != before + 1
                or self.native._geometry != (len(self.ids), self.total)
                or self.native._metadata() is not batch
            ):
                self._issue("native_cold_fast_path_missing")
                return output
            if (depth, layer) in self._seen:
                self._issue("duplicate_depth_layer")
            self._seen.add((depth, layer))
            shape = (len(self.ids), self.total, *q.shape[1:])
            suffix_q = q.reshape(shape)[:, self.prefix :].reshape(-1, *q.shape[1:]).contiguous()
            keys = self.cache.key_cache[:, layer]
            values = self.cache.value_cache[:, layer]
            hot = self.shadow.attend_metadata(suffix_q, keys, values, self._hot)
            if self.shadow._geometry is not None or self.shadow.ragged_calls:
                self._issue("shadow_continuation_path_missing")
                return output
            if new_plan:
                self._check_plan(batch, depth, layer)
            cold_suffix = output.reshape(shape)[:, self.prefix :]
            self._compare_outputs(depth, layer, cold_suffix, hot.reshape_as(cold_suffix))
            self.checked += 1
            return output

        self._patch = patch.object(self.cache, "_attend_prepared", observe)
        self._patch.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._patch.stop()
        self._patch = self._cold = self._hot = None
        if exc_type is None and (len(self._seen) != self.expected or self.checked != self.expected):
            self._issue("incomplete_depth_layer_coverage")
        self._done = exc_type is None

    def report(self):
        if not self._done:
            raise RuntimeError("close a successful frozen attention probe first")
        return dict(
            checked_calls=self.checked,
            expected_calls=self.expected,
            control_issues=self._issues,
            first_gather_mismatch=self._first_gather,
            first_output_diff=self._first_output,
            per_request=self._requests,
        )
