"""Scoped Ouro K/V projection interventions for prefix-cache benchmarks."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn


def _project(forward, rows, counts, input):
    if input.ndim != 2 or input.dtype != torch.bfloat16:
        raise ValueError("fixed K/V projection requires packed 2D BF16 inputs")
    counts["calls"] += 1
    counts["input_rows"] += input.shape[0]
    if rows is None:
        counts["tiles"] += 1
        return forward(input)
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


def _fp32_linear(weight, bias, input):
    with torch.autocast(device_type=input.device.type, enabled=False):
        return F.linear(input.float(), weight, bias).to(input.dtype)


def _without_hf32(matmul, forward, input):
    # The benchmark is single-stream. Keep the caller's math mode for every
    # other projection, even when a selected projection raises.
    previous = matmul.allow_hf32
    matmul.allow_hf32 = False
    try:
        return forward(input)
    finally:
        matmul.allow_hf32 = previous


@contextmanager
def fp32_kv_projections(model, projection="kv", layers=None, chunk_rows=None):
    """Run only selected K/V linears in FP32 and return BF16 to attention."""
    names = {"k": ("k_proj",), "v": ("v_proj",), "kv": ("k_proj", "v_proj")}
    if projection not in names:
        raise ValueError("projection must be k, v or kv")
    if chunk_rows is not None and (type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError("chunk_rows must be a positive integer")
    all_layers = model.model.layers
    indices = list(range(len(all_layers))) if layers is None else list(layers)
    if not indices or any(type(i) is not int or not 0 <= i < len(all_layers) for i in indices):
        raise ValueError(f"choose at least one layer in [0, {len(all_layers) - 1}]")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate layer indices")
    work = {}
    with ExitStack() as hooks:
        for i in indices:
            for name in names[projection]:
                module = getattr(all_layers[i].self_attn, name)
                if not isinstance(module, nn.Linear) or module.weight.dtype != torch.bfloat16:
                    raise ValueError("expected a BF16 Ouro linear projection")
                weight = module.weight.detach().float()
                bias = None if module.bias is None else module.bias.detach().float()
                counts = work[f"layers.{i}.{name}"] = Counter()
                forward = partial(_project, partial(_fp32_linear, weight, bias), chunk_rows, counts)
                if weight.device.type == "npu":
                    matmul = getattr(torch.npu, "matmul", None)
                    if getattr(matmul, "allow_hf32", False):
                        forward = partial(_without_hf32, matmul, forward)
                hooks.enter_context(patch.object(module, "forward", forward))
        yield work
