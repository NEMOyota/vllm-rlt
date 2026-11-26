"""Bounded observation on top of the pinned matrix.trial; no change to sampling or allocation."""

import hashlib
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F
from common import write
from replay_op import pack


def fingerprint(x):
    return hashlib.sha256(
        x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class Probes:
    def __init__(self, model, spec, inputs, mode, folder, selection=None, window=None):
        self.model, self.spec, self.mode, self.folder = model, spec, mode, folder
        self.rid = spec["target"]
        self.prefix = spec["hits"][int(self.rid.split("-")[-1])]
        self.last = len(inputs[2][self.rid]) - 1
        self.selection = selection or {}
        self.events = []
        self.calls = []
        self.layouts = []
        self.counts = Counter()
        self.sdpa = Counter()
        self.layer = None
        self.frame = None
        self.linear = None
        self.head = {}
        self.saved = False
        self.window = window
        self.opened = False

    def scope(self, runner, batch):
        spans = batch.spans
        target = next((s for s in spans if s.request.request_id == self.rid), None)
        if target is None:
            return None
        step = 1 + len(target.request.generated_token_ids)
        if step > self.spec["step"]:
            return None
        rows = []
        for s in spans:
            rows.extend(
                (s.request.request_id, p)
                for p in range(s.token_start, s.token_start + s.token_count)
            )
        return dict(step=step, phase=batch.operation.value, depth=-1, rows=rows)

    def coords(self):
        rows = self.frame["rows"]
        return [
            (i, pos)
            for i, (rid, pos) in enumerate(rows)
            if rid == self.rid
            and (
                pos in (self.prefix, self.last)
                or self.frame["step"] > 1
                or self.frame["phase"] == "coda"
            )
        ]

    def call(self, name):
        f = self.frame
        key = f"{f['step']}/{f['phase']}/d{f['depth']}/{name}"
        self.counts[key] += 1
        return f"{key}/{self.counts[key]}"

    def record(self, key, kind, values, offset=0):
        if self.mode != "scan":
            return
        for row, pos in self.coords():
            i = row - offset
            if 0 <= i < len(values):
                self.events.append(
                    dict(
                        call=key,
                        kind=kind,
                        pos=pos,
                        row=i,
                        shape=list(values.shape),
                        dtype=str(values.dtype),
                        digest=fingerprint(values[i]),
                    )
                )

    def save_head(self, inp, out):
        step = self.frame["step"]
        if step != self.spec["step"] or self.frame["phase"] != "coda":
            return
        idx = next(i for i, (rid, _) in enumerate(self.frame["rows"]) if rid == self.rid)
        hidden = inp[idx].detach().cpu().clone()
        logits = out[idx].detach().cpu().clone()
        torch.save(dict(hidden=hidden, logits=logits), self.folder / "head.pt")
        pair = self.spec["pair"]
        self.head = dict(
            step=step,
            hidden_dtype=str(hidden.dtype),
            logits_dtype=str(logits.dtype),
            pair=pair,
            scores=logits[pair].float().tolist(),
            max_score=float(logits.max()),
            margin_other_minus_off=float(logits[pair[1]].float() - logits[pair[0]].float()),
            file="head.pt",
        )

    @contextmanager
    def installed(self):
        from vllm_rlt.core.kv_cache_manager import KVCacheManager
        from vllm_rlt.models.ouro import OuroForCausalLM
        from vllm_rlt.worker.model_runner import ModelRunner

        execute = ModelRunner._execute_batch
        recur = OuroForCausalLM.recurrent_prepared
        put = KVCacheManager._write_prepared
        att = KVCacheManager._attend_prepared
        sdpa = F.scaled_dot_product_attention
        linear = F.linear

        def run(runner, batch, prepared):
            self.frame = self.scope(runner, batch)
            if self.frame and self.window and not self.opened:
                self.window.start()
                self.opened = True
            try:
                return execute(runner, batch, prepared)
            finally:
                if (
                    self.frame
                    and self.window
                    and self.frame["phase"] == "coda"
                    and self.frame["step"] == self.spec["step"]
                ):
                    self.window.stop()
                    self.window = None
                self.frame = None

        def recurrent(model, hidden, metadata, cache, **kw):
            if not self.frame:
                return recur(model, hidden, metadata, cache, **kw)
            f = self.frame.copy()
            owners = {id(a): rid for rid, a in cache._allocations.items()}
            rows = [(owners[id(a)], pos) for a, d, pos in metadata.rows]
            depths = {d for a, d, pos in metadata.rows}
            if len(depths) != 1 or len(rows) != len(hidden):
                raise ValueError("unexpected padding or mixed depths in synchronous diagnostic")
            self.frame.update(rows=rows, depth=next(iter(depths)))
            if self.mode in ("scan", "profile"):
                spans = []
                for i, (rid, pos) in enumerate(rows):
                    if not spans or spans[-1]["rid"] != rid or spans[-1]["abs_end"] != pos:
                        spans.append(dict(rid=rid, abs_start=pos, row_start=i))
                    spans[-1].update(row_end=i + 1, abs_end=pos + 1)
                self.layouts.append(
                    dict(step=f["step"], phase=f["phase"], depth=self.frame["depth"], spans=spans)
                )
            try:
                return recur(model, hidden, metadata, cache, **kw)
            finally:
                self.frame = f

        def write_kv(cache, layer, metadata, k, v, *a, **kw):
            out = put(cache, layer, metadata, k, v, *a, **kw)
            if self.frame and self.mode == "scan":
                key = self.call(f"layers.{layer}.write")
                self.record(key, "rope_K", k)
                self.record(key, "value", v)
            return out

        def attention(cache, layer, metadata, q, *a, **kw):
            if not self.frame:
                return att(cache, layer, metadata, q, *a, **kw)
            self.layer = layer
            key = self.call(f"layers.{layer}.attention")
            self.record(key, "rope_Q", q)
            if self.prefix and self.frame["step"] == 1 and self.mode == "scan":
                # In reuse these are the actual donor-backed KV rows; in off
                # the same positions have just been computed for the target.
                k, v = cache.read(layer, self.rid, self.frame["depth"], self.prefix)
                for tag, x in (("prefix_K", k), ("prefix_V", v)):
                    self.events.append(
                        dict(
                            call=key,
                            kind=tag,
                            pos=self.prefix - 1,
                            range=[0, self.prefix],
                            digest=fingerprint(x),
                            shape=list(x.shape),
                            dtype=str(x.dtype),
                        )
                    )
            try:
                out = att(cache, layer, metadata, q, *a, **kw)
                if self.mode in ("scan", "profile"):
                    kernel = cache.attention.kernel
                    members = list(dict.fromkeys(rid for rid, pos in self.frame["rows"]))
                    self.calls.append(
                        dict(
                            kind="attention_plan",
                            call=key,
                            request_order=members,
                            direct_decode=kernel._direct_decode,
                            batched_geometry=kernel._geometry,
                            groups=[
                                dict(
                                    sequence=g[0],
                                    rid=members[g[0]],
                                    rows=g[1].numel(),
                                    initial=g[4],
                                )
                                for g in (kernel._groups or ())
                            ],
                        )
                    )
                self.record(key, "attention_output", out)
                return out
            finally:
                self.layer = None

        def observed_sdpa(q, k, v, *a, **kw):
            out = sdpa(q, k, v, *a, **kw)
            if self.frame:
                self.sdpa[str((q.dtype, k.dtype, v.dtype, out.dtype))] += 1
                if self.mode in ("profile", "scan"):
                    self.calls.append(
                        dict(
                            kind="sdpa",
                            step=self.frame["step"],
                            depth=self.frame["depth"],
                            layer=self.layer,
                            q=list(q.shape),
                            k=list(k.shape),
                            v=list(v.shape),
                            causal=bool(kw.get("is_causal")),
                            mask=kw.get("attn_mask") is not None,
                        )
                    )
            return out

        def enter(name):
            def start(module, args):
                if self.frame:
                    self.linear = dict(key=self.call(name), offset=0, tile=0)

            return start

        def leave(name):
            def end(module, args, out):
                if self.frame and name == "lm_head" and self.mode in ("scan", "dump"):
                    self.save_head(args[0], out)
                self.linear = None

            return end

        def observed_linear(x, w, bias=None):
            if not self.linear or not self.frame:
                return linear(x, w, bias)
            state = self.linear
            key, offset = state["key"], state["offset"]
            tile = state["tile"]
            label = f"rlt/{key}/tile{tile}"
            mstx = getattr(torch.npu, "mstx", None) if self.mode == "profile" else None
            stream = torch.npu.current_stream() if mstx else None
            start = time.time_ns()
            token = mstx.range_start(label, stream) if mstx else None
            try:
                out = linear(x, w, bias)
            finally:
                if token is not None:
                    mstx.range_end(token)
            stop = time.time_ns()
            if x.ndim != 2:
                raise ValueError("unexpected non-matrix Linear input")
            if self.mode in ("scan", "profile"):
                self.calls.append(
                    dict(
                        kind="linear",
                        call=key,
                        label=label,
                        tile=tile,
                        offset=offset,
                        x=list(x.shape),
                        w=list(w.shape),
                        dtype=str(x.dtype),
                        weight_dtype=str(w.dtype),
                        output_dtype=str(out.dtype),
                        host_start_ns=start,
                        host_end_ns=stop,
                        mstx=bool(mstx),
                        target_rows=[
                            (pos, i - offset)
                            for i, pos in self.coords()
                            if 0 <= i - offset < len(x)
                        ],
                    )
                )
            self.record(key, "linear_input", x, offset)
            self.record(key, "linear_output", out, offset)
            if self.mode == "dump" and key == self.selection["call"]:
                for i, pos in self.coords():
                    if pos == self.selection["pos"] and 0 <= i - offset < len(x):
                        side = self.selection["sides"][self.folder.name]
                        if (
                            fingerprint(x[i - offset]) != side["input"]
                            or fingerprint(out[i - offset]) != side["output"]
                        ):
                            raise RuntimeError("full dump does not match selected scan row")
                        torch.save(
                            dict(
                                call=key,
                                request=self.rid,
                                pos=pos,
                                row=i - offset,
                                tile=tile,
                                label=label,
                                boundary="F.linear(input,weight,bias); Y=input @ weight.T + bias",
                                transpose_input=False,
                                transpose_weight=True,
                                input=pack(x),
                                weight=pack(w),
                                bias=pack(bias),
                                output=pack(out),
                            ),
                            self.folder / "operator.pt",
                        )
                        self.saved = True
            state.update(offset=offset + len(x), tile=tile + 1)
            return out

        with ExitStack() as stack:
            stack.enter_context(patch.object(ModelRunner, "_execute_batch", run))
            stack.enter_context(patch.object(OuroForCausalLM, "recurrent_prepared", recurrent))
            stack.enter_context(patch.object(KVCacheManager, "_write_prepared", write_kv))
            stack.enter_context(patch.object(KVCacheManager, "_attend_prepared", attention))
            stack.enter_context(patch.object(F, "scaled_dot_product_attention", observed_sdpa))
            stack.enter_context(patch.object(F, "linear", observed_linear))
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    stack.enter_context(module.register_forward_pre_hook(enter(name)))
                    stack.enter_context(module.register_forward_hook(leave(name)))
                elif self.mode == "scan" and (
                    name.endswith("norm") or "layernorm" in name or name.endswith("embed_tokens")
                ):

                    def observe(m, args, out, n=name):
                        if self.frame:
                            self.record(self.call(n), "stage_output", out)

                    stack.enter_context(module.register_forward_hook(observe))
            try:
                yield
            finally:
                if self.window and self.opened:
                    self.window.stop()
                write(
                    self.folder / "observations.json",
                    dict(
                        events=self.events,
                        calls=self.calls,
                        layouts=self.layouts,
                        sdpa_dtypes=dict(self.sdpa),
                        head=self.head,
                        operator_saved=self.saved,
                    ),
                )
