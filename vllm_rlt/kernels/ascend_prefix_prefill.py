"""Plan bounded SDPA sub-batches from one validated packed prefill descriptor."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby, product

import torch

# Bound dense score padding and temporary inputs independently. The memory
# estimate includes Q/output, KV gathers, GQA expansion and a bool mask. SDPA
# implementation workspace is device-dependent.
_MAX_INPUT_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class Span:
    table: int
    offset: int
    depth: int
    start: int
    width: int

    @property
    def context(self):
        return self.start + self.width


def packed_spans(rows):
    offset = 0
    for table, ((_, depth), chunk) in enumerate(groupby(rows, key=lambda r: (id(r[0]), r[1]))):
        start = rows[offset][2]
        width = sum(1 for _ in chunk)
        yield Span(table, offset, depth, start, width)
        offset += width


def partition(spans, slack, *, q_row_bytes, kv_row_bytes):
    """Greedy best fit within a depth, with at most 1/8 extra Q x KV cells."""
    singles, merged = [], []
    ordered = sorted(spans, key=lambda s: (s.depth, s.width, s.context, s.table))
    band = slack + 1
    for _, depth_spans in groupby(ordered, key=lambda s: s.depth):
        buckets = []
        anchors = defaultdict(list)
        for span in depth_spans:
            if span.width == 1:
                singles.append(span)
                continue
            best = None
            work = span.width * span.context
            # The first member is an anchor. Search neighbouring bands too so
            # two shapes on opposite sides of a band boundary can still pair.
            nearby = product(
                range((span.width - slack) // band, span.width // band + 1),
                range((span.context - slack) // band, (span.context + slack) // band + 1),
            )
            for i in (index for key in nearby for index in anchors.get(key, ())):
                members, lo, hi, useful = buckets[i]
                if span.width - members[0].width > slack:
                    continue
                low, high = min(lo, span.context), max(hi, span.context)
                if high - low > slack:
                    continue
                size = len(members) + 1
                cells = size * span.width * high
                extra = cells - (useful + work)
                memory = cells + size * (
                    3 * span.width * q_row_bytes + 2 * high * (kv_row_bytes + q_row_bytes)
                )
                if 8 * extra <= useful + work and memory <= _MAX_INPUT_BYTES:
                    previous_extra = len(members) * members[-1].width * hi - useful
                    candidate = (extra - previous_extra, i, low, high)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                anchors[span.width // band, span.context // band].append(len(buckets))
                buckets.append(([span], span.context, span.context, work))
            else:
                _, index, low, high = best
                members, _, _, useful = buckets[index]
                members.append(span)
                buckets[index] = (members, low, high, useful + work)
        for members, _, _, _ in buckets:
            if len(members) > 1:
                merged.append(members)
            else:
                singles.extend(members)
    return merged, singles


@dataclass
class PrefixGroup:
    size: int
    width: int
    query_rows: torch.Tensor
    slots: torch.Tensor
    mask: torch.Tensor | None
    keep: torch.Tensor | None
    output_rows: torch.Tensor
    work: dict


def _layout(members, device, block_size):
    widths = [s.width for s in members]
    contexts = [s.context for s in members]
    starts = [s.start for s in members]
    width, context = max(widths), max(contexts)
    qi = torch.arange(width, device=device)
    ki = torch.arange(context, device=device)
    offsets = torch.tensor([s.offset for s in members], device=device)[:, None]
    table_rows = torch.tensor([s.table for s in members], device=device)[:, None]
    valid_q = qi
    keep = None
    if min(widths) != width:
        valid_q = torch.minimum(qi, torch.tensor(widths, device=device)[:, None] - 1)
        # Host-known compact indices avoid boolean indexing/nonzero on the device.
        keep = torch.tensor(
            [b * width + j for b, w in enumerate(widths) for j in range(w)], device=device
        )
    query_rows = (offsets + valid_q).flatten()
    output_rows = query_rows if keep is None else query_rows.index_select(0, keep)
    valid_k = ki
    if min(contexts) != context:
        valid_k = torch.minimum(ki, torch.tensor(contexts, device=device)[:, None] - 1)
    mask = None  # All cold: SDPA's square causal mask covers every valid query.
    if min(starts) != max(starts):
        absolute = torch.tensor(starts, device=device)[:, None] + valid_q
        mask = (ki <= absolute[:, :, None]).unsqueeze(1)
    elif starts[0]:
        # Shared start: even the last real query ends inside its own context.
        mask = ki <= (starts[0] + qi[:, None])
    return (
        query_rows,
        keep,
        output_rows,
        table_rows,
        valid_k // block_size,
        valid_k % block_size,
        mask,
    )


def prepare_group(members, tables, block_size, rows_per_block, cached_layout, stream):
    """Reuse shape indices across loops, bind the current descriptor's physical pages."""
    shape = tuple((s.table, s.offset, s.start, s.width) for s in members)
    query_rows, keep, output_rows, table_rows, pages, offsets, mask = cached_layout(
        ("prefix", tables.device, stream, block_size, shape),
        lambda: _layout(members, tables.device, block_size),
    )
    blocks = tables[table_rows, pages].long()
    slots = (blocks * rows_per_block + offsets).flatten()
    widths = [s.width for s in members]
    contexts = [s.context for s in members]
    size, width, context = len(members), max(widths), max(contexts)
    useful = sum(s.width * s.context for s in members)
    extra = size * width * context - useful
    work = dict(
        groups=1,
        padded_groups=int(extra > 0),
        saved_sdpa_calls=size - 1,
        real_query_rows=sum(widths),
        extra_query_rows=size * width - sum(widths),
        extra_kv_rows=size * context - sum(contexts),
        useful_score_cells=useful,
        extra_score_cells=extra,
    )
    return PrefixGroup(size, width, query_rows, slots, mask, keep, output_rows, work)
