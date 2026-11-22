"""Select one linear precision for every arm of a synchronous Ouro experiment."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


def precision_settings(rows):
    return dict(
        precision="native_bf16" if rows is None else "linear_fp32_padded", linear_fp32_chunk=rows
    )


def add_precision_arg(parser):
    def positive(value):
        n = int(value)
        if n < 1:
            raise ValueError("chunk rows must be positive")
        return n

    parser.add_argument(
        "--linear-fp32-chunk",
        metavar="ROWS",
        type=positive,
        help="Run only this precision: FP32 Linear on fixed padded rows, BF16 KV and SDPA",
    )


def linear_usage(work):
    names, calls = work
    return dict(
        installed=len(names),
        executed=len(calls),
        calls=calls.total(),
        uncalled=sorted(set(names) - calls.keys()),
    )


def check_linear_usage(usage):
    # Fixed-depth Ouro may skip the exit gate.
    if usage["executed"] < 1 or set(usage["uncalled"]) - {"model.early_exit_gate"}:
        raise AssertionError(f"FP32 linears did not run on the expected modules: {usage}")


@contextmanager
def _linear_math(matmul, device_type):
    # Keep the surrounding SDPA and non-linear operations in their original math mode.
    previous = getattr(matmul, "allow_hf32", None)
    if previous:
        matmul.allow_hf32 = False
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        if previous:
            matmul.allow_hf32 = previous


def _padded_linear(weight, bias, rows, matmul, calls, name, input):
    if input.dtype != torch.bfloat16 or input.device != weight.device:
        raise ValueError(f"{name} expected BF16 input on {weight.device}")
    if not input.ndim or input.shape[-1] != weight.shape[1]:
        raise ValueError(f"{name} expected {weight.shape[1]} input features")
    flat = input.reshape(-1, weight.shape[1])
    outputs = []
    with _linear_math(matmul, input.device.type):
        for start in range(0, len(flat), rows):
            tile = flat[start : start + rows].float()
            active = len(tile)
            if active < rows:
                tile = F.pad(tile, (0, 0, 0, rows - active))
            projected = F.linear(tile.contiguous(), weight, bias)
            outputs.append(projected[:active].to(input.dtype))
    calls[name] += 1
    if not outputs:
        return input.new_empty((*input.shape[:-1], weight.shape[0]))
    result = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    return result.reshape(*input.shape[:-1], weight.shape[0])


@contextmanager
def padded_fp32_linears(model, rows):
    """Install on each model Linear, including the exit gate and LM head, then restore."""
    if rows is None:
        yield None
        return
    if type(rows) is not int or rows < 1:
        raise ValueError("linear FP32 chunk size must be a positive number of rows")
    names = []
    calls = Counter()
    with ExitStack() as stack:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module.weight.dtype != torch.bfloat16:
                raise ValueError(f"{name} expected BF16 checkpoint weights")
            weight = module.weight.detach().float()
            bias = module.bias.detach().float() if module.bias is not None else None
            matmul = getattr(torch.npu, "matmul", None) if weight.device.type == "npu" else None
            forward = partial(_padded_linear, weight, bias, rows, matmul, calls, name)
            stack.enter_context(patch.object(module, "forward", forward))
            names.append(name)
        if not names:
            raise ValueError("model contains no nn.Linear")
        yield names, calls
