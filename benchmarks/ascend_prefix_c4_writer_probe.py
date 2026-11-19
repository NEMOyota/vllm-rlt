"""Observe the first real cold C4 layer-0 projection, write addresses, and logical KV readback."""

from contextlib import ExitStack
from unittest.mock import patch

import torch


class FirstWriteProbe:
    """Install only for the synchronous cold target burst; retain BF16 prefix rows on CPU."""

    def __init__(self, model, cache, ids, prefix, total):
        self.ids = tuple(ids)
        if len(self.ids) != 4 or len(set(self.ids)) != 4 or not 0 < prefix < total:
            raise ValueError("expected four distinct cold targets with a prefix and suffix")
        if cache.layout != "last_exited" or cache.dtype != torch.bfloat16:
            raise ValueError("expected a BF16 last_exited cache")
        self.prefix = prefix
        self.total = total
        self.cache = cache
        self.layer = model.model.layers[0]
        self.stages = {}
        self.address_issue = None
        self._batch = None
        self._done = False
        self._stack = None

    def _prefix_rows(self, tensor):
        count = self.total * len(self.ids)
        if tensor.ndim < 2 or tensor.shape[0] != count or tensor.dtype != torch.bfloat16:
            raise ValueError(f"unexpected packed BF16 shape {tensor.shape} for {count} real rows")
        return torch.stack(
            [
                tensor[start : start + self.prefix]
                .detach()
                .to(device="cpu", copy=True)
                .reshape(self.prefix, -1)
                for start in range(0, count, self.total)
            ]
        )

    def _begin_layer(self, module, inputs):
        if self._done:
            return
        _, _, batch, cache = inputs
        if self._batch is not None or cache is not self.cache or len(batch.rows) != self.total * 4:
            raise RuntimeError("first layer did not receive a single complete cold C4 batch")
        allocations = dict(batch.allocations)
        if len(batch.allocations) != len(self.ids) or set(allocations) != set(self.ids):
            raise RuntimeError("first cold layer request IDs do not match target burst")
        for slot, rid in enumerate(self.ids):
            live = self.cache._get_allocation(rid)
            if allocations[rid] is not live:
                raise RuntimeError(f"first cold layer has a stale allocation for {rid}")
            rows = batch.rows[slot * self.total : (slot + 1) * self.total]
            if any(
                a is not live or depth != 0 or pos != i for i, (a, depth, pos) in enumerate(rows)
            ):
                raise RuntimeError(
                    f"unexpected first prefill row order, position or depth for {rid}"
                )
        self._batch = batch

    def _capture(self, key, tensor):
        if self._done:
            return
        if self._batch is None or key in self.stages:
            raise RuntimeError(f"unexpected first layer observation: {key}")
        self.stages[key] = self._prefix_rows(tensor)

    def _check_addresses(self, indices, positions=None):
        # Every row in the *whole* cold batch must have its own physical write slot,
        # including suffix rows that could otherwise overwrite a captured prefix.
        count = 4 * self.total
        if len(indices) != count or (positions is not None and len(positions) != count):
            if self.address_issue is None:
                self.address_issue = dict(
                    kind="address_count",
                    expected=count,
                    actual=len(indices),
                    position_count=len(positions) if positions is not None else None,
                )
            return
        seen = {}
        for slot, rid in enumerate(self.ids):
            table = self.cache.get_block_table(rid, 0)
            for pos in range(self.total):
                row = slot * self.total + pos
                block_index, offset = divmod(pos, self.cache.block_size)
                expected = (table[block_index], 0, offset)
                actual = tuple(indices[row])
                if self.address_issue is None:
                    detail = dict(row=row, request_id=rid, position=pos)
                    if actual != expected:
                        source = "staged_address" if positions is not None else "scatter_index"
                        self.address_issue = dict(
                            kind=source, expected=expected, actual=actual, **detail
                        )
                    elif positions is not None and positions[row] != pos:
                        self.address_issue = dict(
                            kind="position_id", expected=pos, actual=positions[row], **detail
                        )
                    elif actual in seen:
                        self.address_issue = dict(
                            kind="alias", address=actual, other_row=seen[actual], **detail
                        )
                seen[actual] = row

    def __enter__(self):
        if self.cache is None or self._stack is not None:
            raise RuntimeError("use a fresh first-write probe for each replay")
        cache = self.cache
        write = cache._write_prepared

        def watch(layer, batch, k, v):
            if self._done:
                return write(layer, batch, k, v)
            if layer != 0 or self._batch is None or self._batch is not batch:
                raise RuntimeError("first target writer does not match the observed cold layer")
            for key in ("hidden", "norm", "v_input", "v_proj"):
                if key not in self.stages:
                    raise RuntimeError(f"first writer ran before observing {key}")
            self._capture("k_write", k)
            self._capture("v_write", v)
            blocks = batch.write_blocks.detach().cpu().tolist()
            offsets = batch.write_offsets.detach().cpu().tolist()
            positions = batch.position_ids.detach().cpu().tolist()
            if len(blocks) != len(offsets):
                self.address_issue = dict(
                    kind="write_address_lengths", blocks=len(blocks), offsets=len(offsets)
                )
            self._check_addresses([(b, 0, off) for b, off in zip(blocks, offsets)], positions)

            result = write(layer, batch, k, v)
            if cache.device.type == "npu" and cache.backend == "ascend_eager":
                indices = cache._npu_writer._indices[layer].detach().cpu().tolist()
                self._check_addresses(indices)
            keys, values = [], []
            for rid in self.ids:
                key, value = cache.read(layer, rid, 0, self.prefix)
                keys.append(key.detach().to(device="cpu", copy=True).reshape(self.prefix, -1))
                values.append(value.detach().to(device="cpu", copy=True).reshape(self.prefix, -1))
            self.stages.update(k_read=torch.stack(keys), v_read=torch.stack(values))
            self._done = True
            self._batch = None
            return result

        with ExitStack() as setup:
            setup.enter_context(self.layer.register_forward_pre_hook(self._begin_layer))
            for module, before, after in (
                (self.layer.input_layernorm, "hidden", "norm"),
                (self.layer.self_attn.v_proj, "v_input", "v_proj"),
            ):
                setup.enter_context(
                    module.register_forward_pre_hook(
                        lambda _m, args, key=before: self._capture(key, args[0])
                    )
                )
                setup.enter_context(
                    module.register_forward_hook(
                        lambda _m, _args, output, key=after: self._capture(key, output)
                    )
                )
            setup.enter_context(patch.object(cache, "_write_prepared", watch))
            self._stack = setup.pop_all()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stack.close()
        self.cache = self.layer = self._stack = self._batch = None
        if exc_type is None and not self._done:
            raise RuntimeError("did not observe a complete depth-0 layer-0 cold target write")
