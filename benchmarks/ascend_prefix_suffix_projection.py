"""Isolate hot suffix V projection row geometry on a synchronous C4 prefill."""

from contextlib import ExitStack
from functools import partial

import torch


class SuffixVProjection:
    """Capture suffix rows and optionally intervene only in the target prefill V projection."""

    def __init__(self, model, cache, ids, prefix, total, mode, *, reference=None):
        self.ids = tuple(ids)
        if len(self.ids) != 4 or len(set(self.ids)) != 4 or not 0 < prefix < total:
            raise ValueError("expected four distinct targets with a prefix and suffix")
        if mode not in {"off", "reuse", "first", "all", "oracle"}:
            raise ValueError(f"unknown suffix projection mode {mode!r}")
        self.layers = model.model.layers
        if (
            cache.layout != "last_exited"
            or cache.dtype != torch.bfloat16
            or len(self.layers) != cache.num_layers
        ):
            raise ValueError("expected a BF16 last_exited cache matching the model layers")
        self.prefix = prefix
        self.total = total
        self.suffix = total - prefix
        self.start = 0 if mode == "off" else prefix
        self.rows_per_slot = total - self.start
        if mode == "oracle":
            width = cache.num_kv_heads * cache.head_dim
            if (
                not isinstance(reference, torch.Tensor)
                or reference.device.type != "cpu"
                or reference.dtype != torch.bfloat16
                or reference.shape != (4, self.suffix, width)
            ):
                raise ValueError("oracle requires the captured cold CPU BF16 suffix V rows")
        elif reference is not None:
            raise ValueError("only oracle accepts cold reference V")
        self.mode = mode
        self.cache = cache
        self.reference = reference
        self.inputs = self.values = None
        self.expected = cache.max_loops * cache.num_layers
        self.observed = self.padded = self.substituted = 0
        self._stack = self._batch = self._active = None
        self._closed = False

    def _check_batch(self, batch, depth):
        if batch.owner is not self.cache or len(batch.rows) != 4 * self.rows_per_slot:
            raise RuntimeError("unexpected target prefill owner or packed row count")
        if len(batch.allocations) != 4 or set(dict(batch.allocations)) != set(self.ids):
            raise RuntimeError("target prefill IDs do not match the four expected targets")
        self.cache._require_live_batch(batch)
        for slot, rid in enumerate(self.ids):
            begin = slot * self.rows_per_slot
            rows = batch.rows[begin : begin + self.rows_per_slot]
            live = self.cache._get_allocation(rid)
            if any(
                a is not live or d != depth or p != pos
                for pos, (a, d, p) in enumerate(rows, self.start)
            ):
                raise RuntimeError(f"unexpected prefill slot, depth or absolute position for {rid}")

    def _enter_layer(self, index, _, args):
        _, _, batch, cache = args
        if cache is not self.cache or self._active is not None:
            raise RuntimeError("unexpected cache or incomplete target V projection")
        if self.observed == self.expected:
            if index == 0 and any(pos < self.total for _, _, pos in batch.rows):
                raise RuntimeError("another prompt traversal followed the complete target prefill")
            return
        depth, layer = divmod(self.observed, len(self.layers))
        if index != layer:
            raise RuntimeError(f"expected prefill depth {depth}, layer {layer}, saw layer {index}")
        if layer == 0:
            self._check_batch(batch, depth)
            self._batch = batch
        elif batch is not self._batch:
            raise RuntimeError("target prefill descriptor changed inside the same traversal")
        self._active = index

    def _packed(self, tensor, rows):
        if tensor.dtype != torch.bfloat16 or tensor.ndim != 2 or tensor.shape[0] != 4 * rows:
            raise RuntimeError(
                f"unexpected BF16 projection shape {tensor.shape} for {4 * rows} rows"
            )
        return tensor.reshape(4, rows, -1)

    def _suffix_rows(self, tensor, rows):
        return self._packed(tensor, rows)[:, -self.suffix :]

    def _pad_this_call(self):
        return self.mode == "all" or (self.mode == "first" and self.observed == 0)

    def _before_v(self, index, module, args):
        if self.observed == self.expected:
            return
        if self._active != index:
            raise RuntimeError("V projection did not originate in the active prefill layer")
        (hidden,) = args
        packed = self._packed(hidden, self.rows_per_slot)
        if self.observed == 0:
            self.inputs = (
                packed[:, -self.suffix :].detach().to(device="cpu", copy=True).contiguous()
            )
        if self._pad_this_call():
            padded = hidden.new_zeros((4, self.total, hidden.shape[-1]))
            padded[:, self.prefix :].copy_(packed)
            return (padded.flatten(0, 1),)

    def _after_v(self, index, module, args, output):
        if self.observed == self.expected:
            return
        if self._active != index:
            raise RuntimeError("V projection output has no matching active prefill layer")
        padded = self._pad_this_call()
        rows = self.total if padded else self.rows_per_slot
        suffix = self._suffix_rows(output, rows)
        result = None
        if padded:
            result = suffix.reshape(4 * self.suffix, -1).contiguous()
            self.padded += 1
        elif self.mode == "oracle" and self.observed == 0:
            suffix = self.reference.to(device=output.device, copy=True)
            result = suffix.reshape_as(output).contiguous()
            self.substituted += 1
        if self.observed == 0:
            self.values = suffix.detach().to(device="cpu", copy=True).contiguous()
        self.observed += 1
        self._active = None
        return result

    def __enter__(self):
        if self.cache is None or self._stack is not None:
            raise RuntimeError("use a fresh suffix projection probe for each target burst")
        with ExitStack() as stack:
            for index, layer in enumerate(self.layers):
                v = layer.self_attn.v_proj
                if v.weight.dtype != torch.bfloat16:
                    raise ValueError("expected a BF16 target V projection")
                stack.enter_context(
                    layer.register_forward_pre_hook(partial(self._enter_layer, index))
                )
                stack.enter_context(v.register_forward_pre_hook(partial(self._before_v, index)))
                stack.enter_context(v.register_forward_hook(partial(self._after_v, index)))
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stack.close()
        self.cache = self.reference = self._batch = self.layers = self._stack = None
        if exc_type is None:
            expected_pads = {"first": 1, "all": self.expected}.get(self.mode, 0)
            if (
                self.observed != self.expected
                or self.padded != expected_pads
                or self.substituted != int(self.mode == "oracle")
            ):
                raise RuntimeError("incomplete target prefill V projection coverage")
            self._closed = True

    def report(self):
        if not self._closed:
            raise RuntimeError(
                "complete the target replay before reading suffix projection coverage"
            )
        return dict(
            observed_calls=self.observed,
            expected_calls=self.expected,
            modified_calls=self.padded + self.substituted,
            padded_calls=self.padded,
            oracle_calls=self.substituted,
        )
