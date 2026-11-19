"""Benchmark intervention that fixes the BF16 K/V projection matmul row count."""

from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

import torch
from torch import nn


def _project(forward, rows, counts, input):
    if input.ndim != 2 or input.dtype != torch.bfloat16:
        raise ValueError("fixed K/V projection requires packed 2D BF16 inputs")
    counts["calls"] += 1
    counts["input_rows"] += input.shape[0]
    outputs = []
    for tile in input.split(rows, dim=0):
        active = tile.shape[0]
        padding = rows - active
        if padding:
            tile = torch.cat((tile, tile.new_zeros((padding, tile.shape[1]))), dim=0)
        outputs.append(forward(tile.contiguous())[:active])
        counts["tiles"] += 1
        counts["padded_rows"] += padding
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


@contextmanager
def fixed_kv_rows(model, rows):
    """Temporarily tile each Ouro K/V projection; yield aggregate work counters."""
    if type(rows) is not int or rows <= 0:
        raise ValueError("rows must be a positive integer")
    counts = {"calls": 0, "input_rows": 0, "tiles": 0, "padded_rows": 0}
    with ExitStack() as hooks:
        for layer in model.model.layers:
            for projection in (layer.self_attn.k_proj, layer.self_attn.v_proj):
                if (
                    not isinstance(projection, nn.Linear)
                    or projection.weight.dtype != torch.bfloat16
                ):
                    raise ValueError("fixed K/V projection requires BF16 linear weights")
                wrapped = partial(_project, projection.forward, rows, counts)
                hooks.enter_context(patch.object(projection, "forward", wrapped))
        yield counts
