"""Benchmark-only cold/suffix grouping using the mainline packed tables and Torch SDPA."""

from collections import defaultdict
from contextlib import nullcontext
from unittest.mock import patch

import torch

from vllm_rlt.kernels import ascend_attention as ascend


class GroupedAscendAttention(ascend.AscendAttention):
    def __init__(self):
        super().__init__()
        self.cold_prefill_calls = 0
        self.suffix_prefill_calls = 0
        self._members = []
        self._group_plan = None

    def _plan(self, sequence_keys, lengths, device, *, packed=False):
        groups = super()._plan(sequence_keys, lengths, device, packed=packed)
        self._members = []
        self._group_plan = None
        if not packed:
            return groups
        # Use host lengths. Packed page-table indices count sequences, while Q
        # indices count tokens and may be nonadjacent across members of a bucket.
        rows = defaultdict(list)
        for key, length in zip(sequence_keys, lengths, strict=True):
            if length:
                rows[key].append(length)
        by_geometry = defaultdict(list)
        for ((_, depth), ends), group in zip(rows.items(), groups, strict=True):
            start, stop = ends[0] - 1, ends[-1]
            if len(ends) > 1 and ends == list(range(start + 1, stop + 1)):
                by_geometry[depth, start, stop].append(group)
        self._members = [members for members in by_geometry.values() if len(members) >= 2]
        return groups

    def _forward(self, q, keys, values, tables, groups, *, flat_rows=None, result=None):
        if not self._members or flat_rows is None:
            return super()._forward(
                q, keys, values, tables, groups, flat_rows=flat_rows, result=result
            )
        if self._group_plan is None:
            merged = []
            included = set()
            for members in self._members:
                _, indices, positions, mask, initial, _ = members[0]
                page_rows = [group[0] for group in members]
                blocks = tables[
                    torch.tensor(page_rows, device=q.device)[:, None], positions // keys.shape[1]
                ]
                slots = (blocks.long() * flat_rows[2] + positions % keys.shape[1]).flatten()
                query_rows = torch.cat([group[1] for group in members])
                merged.append((len(members), indices.numel(), query_rows, slots, mask, initial))
                included.update(page_rows)
            # The mainline caller handles one-token ragged decode independently.
            self._group_plan = merged, [group for group in groups if group[0] not in included]
        merged, singles = self._group_plan
        result = super()._forward(
            q, keys, values, tables, singles, flat_rows=flat_rows, result=result
        )
        flat_keys, flat_values, _ = flat_rows
        for size, width, rows, slots, mask, initial in merged:
            kv_shape = (size, -1, keys.shape[-2], keys.shape[-1])
            k = flat_keys.index_select(0, slots).reshape(kv_shape).transpose(1, 2)
            v = flat_values.index_select(0, slots).reshape(kv_shape).transpose(1, 2)
            query = q.index_select(0, rows).reshape(size, width, *q.shape[1:])
            output = ascend._attend(query.transpose(1, 2), k, v, mask, is_causal=initial)
            result.index_copy_(0, rows, output.transpose(1, 2).reshape(-1, *q.shape[1:]))
            if initial:
                self.cold_prefill_calls += 1
            else:
                self.suffix_prefill_calls += 1
        return result


def attention_mode(mode):
    if mode == "grouped":
        return patch.object(ascend, "AscendAttention", GroupedAscendAttention)
    if mode == "main":
        return nullcontext()
    raise ValueError(f"unknown attention mode {mode!r}")


def group_counts(kernel):
    return {kind: getattr(kernel, f"{kind}_prefill_calls", 0) for kind in ("cold", "suffix")}
