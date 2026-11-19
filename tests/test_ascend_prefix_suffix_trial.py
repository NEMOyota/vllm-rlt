"""Check hot C4 suffix-projection interventions against the real cold row positions."""

import copy
from contextlib import contextmanager
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_suffix_projection import SuffixVProjection
from benchmarks.ascend_prefix_suffix_trial import suffix_trial
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model


def test_suffix_trial_compares_native_and_candidate_runs_with_matching_archive():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    contract = {"seed": 168, "output_tokens": 3}
    archived = {
        "runs": {mode: replay_mode(model, "torch", case, mode, 3, 168) for mode in ("off", "reuse")}
    }
    result = suffix_trial(model, "torch", case, contract, archived)
    summary, runs = result["summary"], result["runs"]
    assert summary["comparable"] and not summary["controls"]
    traversals = model.config.num_hidden_layers * model.config.total_ut_steps
    ids = [f"target-{i}" for i in range(case.concurrency)]
    for mode, modified in (
        ("off", 0),
        ("reuse", 0),
        ("first", 1),
        ("all", traversals),
        ("oracle", 1),
    ):
        run = runs[mode]
        counters = run["projection"]
        assert counters["observed_calls"] == traversals and counters["modified_calls"] == modified
        assert [r["request_id"] for r in run["requests"]] == ids
        hits = case.prefix if mode != "off" else 0
        assert [r["hit_tokens"] for r in run["requests"]] == [hits] * case.concurrency
        assert (
            run["prefix_hits"] == hits * case.concurrency
            and run["prefix_queries"] == case.concurrency
        )

    for rid in ids:
        assert summary["first_layer"]["oracle"][rid]["v_output_different_elements"] == 0
    for arm, comparison in (
        ("reuse", "native_off_vs_reuse"),
        ("first", "off_vs_first"),
        ("all", "off_vs_all"),
        ("oracle", "off_vs_oracle"),
    ):
        differences = []
        for cold, hot in zip(runs["off"]["requests"], runs[arm]["requests"], strict=True):
            assert cold["prompt_sha256"] == hot["prompt_sha256"]
            for pos, (a, b) in enumerate(zip(cold["tokens"], hot["tokens"], strict=True), 1):
                if a != b:
                    differences.append(
                        {"request_id": cold["request_id"], "position": pos, "ids": [a, b]}
                    )
                    break
        assert differences == summary[comparison]

    stale = copy.deepcopy(archived)
    stale["runs"]["reuse"]["requests"][1]["prefill_start"] = 0
    rejected = suffix_trial(model, "torch", case, contract, stale)["summary"]
    assert not rejected["comparable"] and rejected["controls"]


def test_position_sensitive_projection_keeps_cold_row_indices_only_on_first_prefill_call():
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    ids = [f"target-{i}" for i in range(case.concurrency)]
    prefix, suffix = case.prefix, case.suffix
    total = prefix + suffix
    model = tiny_ouro_model(dtype=torch.bfloat16)
    projection = model.model.layers[0].self_attn.v_proj
    original = projection.forward
    args_seen, probes = [], {}

    def tagged_forward(hidden):
        args_seen.append(hidden.detach().clone())
        output = original(hidden).clone()
        # All packed indices here are exactly representable in BF16. Any misplaced
        # row can be identified in the observed channel independently of the weights.
        output[:, 0] = torch.arange(hidden.shape[0], dtype=output.dtype, device=output.device)
        return output

    @contextmanager
    def capture(engine, targets, donor):
        with patch.object(projection, "forward", tagged_forward):
            probes["suffix"] = SuffixVProjection(
                model, engine.cache_manager, ids, prefix, total, "first"
            )
            with probes["suffix"]:
                yield
            assert projection.forward is tagged_forward
            for layer in model.model.layers:
                assert not layer._forward_pre_hooks
                assert not layer.self_attn.v_proj._forward_pre_hooks
                assert not layer.self_attn.v_proj._forward_hooks

    run = replay_mode(model, "torch", case, "reuse", 3, 168, target_context=capture)
    probe = probes["suffix"]
    cold_rows = torch.arange(total * 4).reshape(4, total)[:, prefix:].to(torch.bfloat16)
    assert torch.equal(probe.values[..., 0], cold_rows)
    padded = args_seen[0].reshape(4, total, -1)
    assert not torch.count_nonzero(padded[:, :prefix])
    assert torch.equal(padded[:, prefix:], probe.inputs)
    traversal_rows = [x.shape[0] for x in args_seen]
    assert traversal_rows[:4] == [4 * total, 4 * suffix, 4 * suffix, 4 * suffix]
    assert traversal_rows[4:] and all(0 < n <= 4 for n in traversal_rows[4:])
    assert [r["hit_tokens"] for r in run["requests"]] == [prefix] * 4
    assert projection.forward == original
