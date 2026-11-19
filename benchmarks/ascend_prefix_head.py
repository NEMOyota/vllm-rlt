"""Inspect one real Ouro LM-head decision and recompute two dot products on CPU."""

import hashlib
import math
from unittest.mock import patch

import torch

from vllm_rlt.request import Operation


def _two_scores(row, ids):
    # Select two scalar views. Keep conversions on CPU, without staging a vocabulary row.
    return [row[token].detach().cpu().item() for token in ids]


class HeadProbe:
    """Observe a request's 1-based generated position during a synchronous replay."""

    def __init__(self, runner, model, rid, position, ids):
        self.runner = runner
        self.head = model.lm_head
        self.rid = rid
        self.position = position
        self.ids = tuple(ids)
        if position < 1 or len(self.ids) != 2 or self.ids[0] == self.ids[1]:
            raise ValueError("a positive position and two distinct vocabulary IDs are required")
        if any(
            type(token) is not int or not 0 <= token < self.head.out_features for token in self.ids
        ):
            raise ValueError("candidate ID outside the LM-head vocabulary")
        if runner.model is not model or self.head.bias is not None:
            raise ValueError("expected the runner's Ouro model with a bias-free LM head")
        self.hidden = None
        self._result = None
        self._patch = None

    def __enter__(self):
        if self._patch is not None or self.runner is None:
            raise RuntimeError("a head probe can observe only one replay context")
        execute = self.runner._execute_batch

        def observe(batch, prepared):
            if batch.operation != Operation.CODA:
                return execute(batch, prepared)
            # output_indices holds a generated position for each *real span*, not a
            # tensor row number. Padded hidden/logit rows follow these real rows.
            matches = [
                i
                for i, (span, index) in enumerate(
                    zip(batch.spans, prepared.output_indices, strict=True)
                )
                if span.request.request_id == self.rid and index + 1 == self.position
            ]
            if not matches:
                return execute(batch, prepared)
            if len(matches) != 1 or self._result is not None:
                raise RuntimeError("ambiguous CODA decision for the requested target and position")
            return self._inspect(execute, batch, prepared, matches[0])

        self._patch = patch.object(self.runner, "_execute_batch", observe)
        self._patch.start()
        return self

    def _inspect(self, execute, batch, prepared, row):
        head_scores = None
        head_dtype = None
        sampler_scores = None
        selected = None
        calls = 0

        def capture_head(module, inputs, output):
            nonlocal head_scores, head_dtype
            (hidden,) = inputs
            if (
                head_scores is not None
                or hidden.ndim != 2
                or output.ndim != 2
                or hidden.shape != (output.shape[0], module.in_features)
                or output.shape[1] != module.out_features
                or output.shape[0] < len(batch.spans)
            ):
                raise RuntimeError("unexpected LM-head call or tensor shape for CODA")
            self.hidden = hidden[row].detach().to(device="cpu", copy=True).contiguous()
            head_scores = _two_scores(output[row], self.ids)
            head_dtype = str(output.dtype)

        sample = self.runner.sampler.sample

        def capture_sample(logits, params, generator):
            nonlocal calls, sampler_scores, selected
            current = calls
            calls += 1
            if current != row:
                return sample(logits, params, generator)
            if params.temperature != 0 or logits.shape != (self.head.out_features,):
                raise RuntimeError("expected one greedy vocabulary row for the target")
            sampler_scores = _two_scores(logits, self.ids)
            token = sample(logits, params, generator)
            selected = int(token.item())
            return token

        with (
            self.head.register_forward_hook(capture_head),
            patch.object(self.runner.sampler, "sample", capture_sample),
        ):
            result = execute(batch, prepared)
        if calls != len(batch.spans) or head_scores is None or sampler_scores is None:
            raise RuntimeError("LM-head observation and sampled CODA rows are incomplete")
        if head_scores != sampler_scores or not all(math.isfinite(s) for s in head_scores):
            raise RuntimeError(
                f"invalid LM-head/sampler correspondence: {head_scores} vs {sampler_scores}"
            )

        # Read only two checkpoint weight rows. CPU casts of the original hidden
        # and stored weights preserve their values exactly before FP64 accumulation.
        h64 = self.hidden.double()
        scores64 = [
            torch.dot(self.head.weight[token].detach().cpu().double(), h64).item()
            for token in self.ids
        ]
        if not all(math.isfinite(score) for score in scores64):
            raise RuntimeError("nonfinite reference score for the captured LM-head row")
        self._result = dict(
            request_id=self.rid,
            position=self.position,
            ids=list(self.ids),
            head_logits=head_scores,
            sampler_logits=sampler_scores,
            fp64_logits=scores64,
            hidden_sha256=hashlib.sha256(self.hidden.view(torch.uint8).numpy()).hexdigest(),
            hidden_dtype=str(self.hidden.dtype),
            weight_dtype=str(self.head.weight.dtype),
            head_dtype=head_dtype,
            selected_id=selected,
        )
        return result

    def __exit__(self, exc_type, exc, tb):
        self._patch.stop()
        self._patch = None
        self.head = self.runner = None

    def report(self):
        if self._result is None:
            raise RuntimeError(f"no LM-head decision for {self.rid} at position {self.position}")
        return self._result
