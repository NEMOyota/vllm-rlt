"""Target-only instrumentation for the single-request Ascend prefix experiment."""

import math
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.ascend_prefix_combo_checks import require
from vllm_rlt.kernels import ascend_attention as ascend


class PrefillOps:
    """Count padded GEMMs and time selected operations on their own stream."""

    def __init__(self, device, chunk, length, hit, *, time_q_pad=False):
        self.device = device
        self.chunk = chunk
        self.length = length
        self.width = length - hit
        self.time_q_pad = time_q_pad
        self.flat = None
        self.active = None
        self.forwards = Counter()
        self.tiles = Counter()
        self.spans = defaultdict(list)
        self.bytes = Counter()

    def _measure(self, name, operation, *args, **kwargs):
        if self.device.type == "npu":
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            result = operation(*args, **kwargs)
            end.record()
        else:  # CPU clock only for the host validation fixture.
            begin = time.perf_counter_ns()
            result = operation(*args, **kwargs)
            end = time.perf_counter_ns()
        self.spans[name].append((begin, end))
        return result

    def _forward(self, name, forward, input):
        previous = self.active
        self.active = (name, input.numel() // input.shape[-1])
        self.forwards[self.active] += 1
        try:
            return forward(input)
        finally:
            self.active = previous

    @contextmanager
    def observe(self, model):
        flat_rows, select = ascend._flat_kv_rows, torch.Tensor.index_select
        linear, sdpa, native_pad = F.linear, F.scaled_dot_product_attention, F.pad

        def flat(*args):
            self.flat = flat_rows(*args)
            return self.flat

        def gather(tensor, dim, index):
            # Track the exact two per-layer views used by both packed paths. Q,
            # page-table and other index_select calls are left out.
            if self.flat is not None and (tensor is self.flat[0] or tensor is self.flat[1]):
                kind = "k_read" if tensor is self.flat[0] else "v_read"
                require(dim == 0 and index.numel() == self.length, "wrong KV read width")
                result = self._measure(kind, select, tensor, dim, index)
                self.bytes[kind] += result.numel() * result.element_size()
                return result
            return select(tensor, dim, index)

        def project(input, weight, bias=None):
            require(
                self.active is not None
                and input.ndim == 2
                and input.shape[0] == self.chunk
                and input.dtype == weight.dtype == torch.float32,
                "unexpected projection or tile outside the padded FP32 linear",
            )
            self.tiles[self.active] += 1
            return linear(input, weight, bias)

        def pad_tile(input, pad, *args, **kwargs):
            if self.active is not None and self.active[0] == "q_proj":
                require(pad == (0, 0, 0, self.chunk - len(input)), "unexpected Q padding")
                return self._measure("q_pad", native_pad, input, pad, *args, **kwargs)
            return native_pad(input, pad, *args, **kwargs)

        def attention(query, key, value, **kwargs):
            mask = kwargs.get("attn_mask")
            causal = kwargs.get("is_causal", False)
            require(
                query.shape[0] == 1
                and query.shape[-2] == self.width
                and key.shape[-2] == value.shape[-2] == self.length
                and query.dtype == key.dtype == value.dtype == torch.bfloat16,
                "unexpected SDPA geometry or dtype",
            )
            if self.width == self.length:
                require(causal and mask is None, "expected full-prompt causal SDPA")
            else:
                require(
                    not causal
                    and mask is not None
                    and mask.dtype == torch.bool
                    and tuple(mask.shape[-2:]) == (self.width, self.length),
                    "expected an explicit suffix causal mask",
                )
            return self._measure("sdpa", sdpa, query, key, value, **kwargs)

        with ExitStack() as stack:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    family = name.rsplit(".", 1)[-1]
                    stack.enter_context(
                        patch.object(
                            module, "forward", partial(self._forward, family, module.forward)
                        )
                    )
            stack.enter_context(patch.object(F, "linear", project))
            if self.time_q_pad:
                stack.enter_context(patch.object(F, "pad", pad_tile))
            stack.enter_context(patch.object(ascend, "_flat_kv_rows", flat))
            stack.enter_context(patch.object(torch.Tensor, "index_select", gather))
            stack.enter_context(patch.object(F, "scaled_dot_product_attention", attention))
            yield

    def result(self, expected_calls):
        require(
            self.forwards["q_proj", self.width] == expected_calls
            and self.forwards.keys() == self.tiles.keys()
            and all(
                self.tiles[key] == calls * math.ceil(key[1] / self.chunk)
                for key, calls in self.forwards.items()
            ),
            "actual linear tile counts disagree with target rows",
        )
        for kind in ("k_read", "v_read", "sdpa"):
            require(len(self.spans[kind]) == expected_calls, f"missing {kind} calls")
        if self.time_q_pad:
            expected_pads = expected_calls if self.width % self.chunk else 0
            require(len(self.spans["q_pad"]) == expected_pads, "unexpected number of Q pad calls")
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)
            elapsed = {
                k: sum(a.elapsed_time(b) for a, b in spans) for k, spans in self.spans.items()
            }
        else:
            elapsed = {k: sum(b - a for a, b in spans) / 1e6 for k, spans in self.spans.items()}
        require(all(math.isfinite(ms) and ms >= 0 for ms in elapsed.values()), "invalid op timer")
        return dict(
            clock="npu_stream_event_ms" if self.device.type == "npu" else "cpu_wall_ms",
            linear_tiles=self.tiles.total(),
            linear_padded_rows=self.tiles.total() * self.chunk
            - sum(rows * calls for (_, rows), calls in self.forwards.items()),
            q_proj_tiles=self.tiles["q_proj", self.width],
            projections=[
                dict(family=name, input_rows=rows, calls=n, tiles=self.tiles[name, rows])
                for (name, rows), n in sorted(self.forwards.items())
            ],
            calls={k: len(v) for k, v in self.spans.items()},
            kv_gather_bytes=dict(self.bytes),
            kv_read_ms=elapsed["k_read"] + elapsed["v_read"],
            sdpa_ms=elapsed["sdpa"],
            op_ms=elapsed,
        )
