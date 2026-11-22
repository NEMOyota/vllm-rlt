"""Isolate cold-prompt and cached-suffix batching in the all-Linear FP32 c4_h2 case."""

import argparse
import math
from dataclasses import dataclass
from functools import partial
from itertools import combinations

import torch

if __package__:
    from . import ascend_prefix_three_plus_one as mixed
else:  # Allow running from an archive next to the server's local FP32 matrix.
    import ascend_prefix_three_plus_one as mixed

paired = mixed.paired


@dataclass
class SuffixPair(mixed.ColdBatch):
    """Cache indices and absolute masks for this metadata, never K/V from another layer."""

    suffix_indices: torch.Tensor | None = None
    read_slots: torch.Tensor | None = None
    masks: torch.Tensor | None = None

    @classmethod
    def plan(cls, batch, groups, cold_count=2):
        plan = super().plan(batch, groups, cold_count)
        if plan is None:
            return None
        first, second = plan.hot_groups
        # Each mask in _plan uses the original context lengths. Require a contiguous
        # suffix at the same absolute positions in both requests before batching.
        for group in (first, second):
            row, indices, positions, *_ = group
            alloc, depth, start = batch.rows[row]
            if positions.numel() != start + indices.numel():
                return None
            query_positions = [p for a, d, p in batch.rows if a is alloc and d == depth]
            if query_positions != list(range(start, positions.numel())):
                return None
        if (first[1].numel(), first[2].numel()) != (second[1].numel(), second[2].numel()):
            return None
        plan.suffix_indices = torch.cat([g[1] for g in plan.hot_groups])
        plan.masks = torch.stack([g[3] for g in plan.hot_groups])[:, None]
        return plan

    def suffixes(self, native, owner, q, keys, values, tables, flat_rows, result):
        flat_keys, flat_values, rows_per_block = flat_rows
        block_size, kv_heads, head_dim = keys.shape[-3:]
        context = self.hot_groups[0][2].numel()
        width = self.hot_groups[0][1].numel()
        if self.read_slots is None:
            positions = self.hot_groups[0][2]
            pages = positions // block_size
            # Each table includes its own physical pages, including private suffix pages.
            blocks = torch.stack([tables[g[0], pages] for g in self.hot_groups])
            self.read_slots = (
                (blocks.long() * rows_per_block + positions % block_size)
                .clamp(0, flat_keys.shape[0] - 1)
                .flatten()
            )

        def gather(flat):
            rows = flat.index_select(0, self.read_slots).reshape(2, context, kv_heads, head_dim)
            heads = rows.transpose(1, 2)
            if q.shape[1] != kv_heads:
                heads = heads.repeat_interleave(q.shape[1] // kv_heads, dim=1)
            return heads.contiguous()

        query = q.index_select(0, self.suffix_indices).reshape(2, width, q.shape[1], head_dim)
        query = query.transpose(1, 2).contiguous()
        key, value = gather(flat_keys), gather(flat_values)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        probs = torch.softmax(scores + self.masks, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probs, value).transpose(1, 2).reshape(-1, q.shape[1], head_dim)
        if result is None:
            result = torch.zeros_like(q)
        result.index_copy_(0, self.suffix_indices, output)
        return result


class TwoPlusTwo(mixed.ThreePlusOne):
    case_name = "c4_h2"
    label = "TWO_PLUS_TWO"
    intermediate_modes = ("cold_pair",)
    candidate = "two_plus_two"
    cold_count = 2
    batched_suffix_mode = "two_plus_two"
    profile_pair = None

    def variant_route(self, mode, audit):
        plan = SuffixPair.plan if mode == self.batched_suffix_mode else mixed.ColdBatch.plan
        return mixed.route(
            mode != "original", audit, cold_count=2, plan_batch=partial(plan, cold_count=2)
        )

    def timing_summary(self):
        measured = self.runs[1:]
        comparisons = {}
        for base, candidate in combinations(self.modes, 2):
            summary = paired.ttft_review(measured, candidate, base)
            requests = {}
            for i, row in enumerate(self.runs[0][base]["reuse"]["requests"]):
                base_times = [r[base]["reuse"]["requests"][i]["ttft_ms"] for r in measured]
                times = [r[candidate]["reuse"]["requests"][i]["ttft_ms"] for r in measured]
                requests[f"target-{i}"] = dict(
                    hit_tokens=row["hit_tokens"],
                    **paired.paired_ms(base_times, times, candidate, base),
                )
            comparisons[f"{base}_vs_{candidate}"] = dict(ttft=summary, reuse_requests=requests)
        return {"comparisons": comparisons}


class ColdPairOnly(TwoPlusTwo):
    intermediate_modes = ()
    candidate = "cold_pair"


if __name__ == "__main__":
    # If the new suffix path fails parity, the independently useful cold pairing
    # can still be measured with exactly the same reference and workload.
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--cold-only", action="store_true")
    mode, rest = mode_parser.parse_known_args()
    paired.main(
        rest,
        pair_type=ColdPairOnly if mode.cold_only else TwoPlusTwo,
        description=__doc__ + " Add --cold-only to run just original and cold_pair.",
    )
