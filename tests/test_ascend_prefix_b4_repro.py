"""Exercise the B4 replay geometries, paired precision and divergence reports on CPU."""

import json

import pytest
import torch

from benchmarks import ascend_prefix_pr1_diagnostics as diagnostics
from tests.test_ascend_main_prefix_combo import cpu_packed_model  # noqa: F401


@pytest.mark.parametrize("cpu_packed_model", [{"max_position_embeddings": 1200}], indirect=True)
def test_b4_replay_preserves_inputs_hits_and_first_different_token(
    tmp_path,
    cpu_packed_model,  # noqa: F811
    monkeypatch,
):
    expected_hits = {
        "p1024_s8": [1024],
        "total512_p384": [384],
        "c4_h3": [128, 128, 128, 0],
        "c4_h4": [128, 128, 128, 128],
    }
    assert set(diagnostics.KNOWN) == set(expected_hits)
    # Deterministic output 0, except at the third target CODA in one native hit arm.
    with torch.no_grad():
        cpu_packed_model.lm_head.weight.zero_()
    trace = diagnostics.trace_logits

    def inject(model, case, inputs, arm, *, linear_work):
        if linear_work is not None or case != diagnostics.matrix.FULL["c4_h3"] or arm != "reuse":
            return trace(model, case, inputs, arm, linear_work=linear_work)
        step = 0

        def change_one_row(_head, _inputs, scores):
            nonlocal step
            if len(scores) == 4:
                step += 1
                if step == 3:
                    scores = scores.clone()
                    scores[0, 1] = 1
            return scores

        with model.lm_head.register_forward_hook(change_one_row):
            return trace(model, case, inputs, arm)

    monkeypatch.setattr(diagnostics, "trace_logits", inject)
    prompts = {}
    for chunk in (None, 256):
        out = tmp_path / str(chunk)
        out.mkdir()
        code = diagnostics.run_drift(cpu_packed_model, diagnostics.KNOWN, out, chunk)
        assert code == (1 if chunk is None else 0)
        review = json.loads((out / "review.json").read_text())
        assert review["differences"] == (1 if chunk is None else 0)
        for name, hits in expected_hits.items():
            row = json.loads((out / f"{name}.json").read_text())
            inputs = (row["primer_ids"], row["target_prompt_ids"])
            assert prompts.setdefault(name, inputs) == inputs
            assert row["linear_fp32_chunk"] == chunk
            assert len(row["target_prompt_ids"]) == row["requests"] == len(hits)
            assert set(row["arms"]) == {"off", "miss", "reuse"}
            for arm, trial in row["arms"].items():
                assert ("linear_work" in trial) == (chunk == 256)
                if chunk == 256:
                    assert trial["linear_work"]["executed"] > 0
                for i, expected in enumerate(hits):
                    rid = f"target-{i}"
                    observed = trial["requests"][rid]
                    assert len(observed["tokens"]) == 16
                    assert observed["hit_tokens"] == (expected if arm == "reuse" else 0)
                    assert (
                        observed["prefill_rows"]
                        == len(row["target_prompt_ids"][rid]) - observed["hit_tokens"]
                    )
    row = json.loads((tmp_path / "None" / "c4_h3.json").read_text())
    assert (row["off_vs_miss"], row["off_vs_reuse"]) == (0, 1)
    (diff,) = row["differences"]
    assert diff["request"] == "target-0" and diff["comparison"] == "off_vs_reuse"
    assert diff["first_diff_1based"] == 3 and diff["prior_generated_ids"] == [0, 0]
    assert (diff["expected"], diff["actual"], diff["cached_tokens"]) == (0, 1, 128)


def test_missing_cache_evidence_stays_incomplete(tmp_path, monkeypatch):
    def run(_model, name, *, linear_work):
        if name == "c4_h3":
            raise AssertionError("unexpected hit count")
        return {"target_prompt_ids": {"target-0": [1]}, "differences": []}

    monkeypatch.setattr(diagnostics, "diagnose_case", run)
    assert diagnostics.run_drift(None, ["c4_h3", "p1024_s8"], tmp_path, None) == 2
    review = json.loads((tmp_path / "review.json").read_text())
    assert review["status"] == "INCOMPLETE" and review["results"]["p1024_s8"]["status"] == "PASS"
    row = json.loads((tmp_path / "c4_h3.json").read_text())
    assert row["status"] == "INCOMPLETE" and "unexpected hit count" in row["detail"]
    assert "off_vs_reuse" not in row
