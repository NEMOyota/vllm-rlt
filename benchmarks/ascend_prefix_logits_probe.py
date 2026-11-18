"""Record the actual greedy sampler inputs during a synchronous diagnostic replay."""

import math
from unittest.mock import patch

import torch

from vllm_rlt.request import Operation


class LogitsProbe:
    """Observe CODA decisions of target-* requests on one ModelRunner.

    Install before the first target is sampled. Host readbacks are intentional
    and belong in a diagnostic replay outside a latency measurement.
    """

    def __init__(self, runner):
        self.runner = runner
        self._rows = {}
        self._hook = None

    def __enter__(self):
        if self._hook is not None:
            raise RuntimeError("logits probe is already active")
        execute = self.runner._execute_batch

        def observe(batch, prepared):
            if batch.operation != Operation.CODA or not any(
                span.request.request_id.startswith("target-") for span in batch.spans
            ):
                return execute(batch, prepared)
            # The runner snapshots one output index per real span. It calls
            # sampler.sample for those spans only, even with padded logits.
            rows = iter(zip(batch.spans, prepared.output_indices, strict=True))
            sample = self.runner.sampler.sample

            def observe_sample(logits, params, generator):
                span, output_index = next(rows)
                rid = span.request.request_id
                if not rid.startswith("target-"):
                    return sample(logits, params, generator)
                if params.temperature != 0 or logits.ndim != 1 or logits.numel() < 2:
                    raise ValueError("expected one greedy vocabulary row with at least two IDs")
                # Copy before sampling, preserving the actual BF16 values.
                # A private copy also permits masking the winner to find the
                # runner-up when the sampler and topk order tied values differently.
                scores = logits.detach().to(device="cpu", dtype=torch.float32, copy=True)
                selected = sample(logits, params, generator)
                self._record(rid, output_index, scores, int(selected.item()))
                return selected

            with patch.object(self.runner.sampler, "sample", observe_sample):
                result = execute(batch, prepared)
            if next(rows, None) is not None:
                raise RuntimeError("CODA emitted fewer sampling decisions than request spans")
            return result

        self._hook = patch.object(self.runner, "_execute_batch", observe)
        self._hook.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._hook.stop()
        self._hook = None

    def _record(self, rid, output_index, scores, selected):
        rows = self._rows.setdefault(rid, [])
        if output_index != len(rows):
            raise RuntimeError(f"noncontiguous generated position {output_index + 1} for {rid}")
        winner = scores[selected].item()
        scores[selected] = -torch.inf
        runner_up = int(scores.argmax().item())
        other = scores[runner_up].item()
        if not math.isfinite(winner) or not math.isfinite(other) or other > winner:
            raise RuntimeError(f"invalid greedy logits for {rid} at position {output_index + 1}")
        rows.append(
            {
                "position": output_index + 1,
                "top_ids": [selected, runner_up],
                "top_logits": [winner, other],
                "margin": winner - other,
                "selected_id": selected,
            }
        )

    def report(self):
        """Return the decisions keyed by request ID in generated-token order."""
        return self._rows
