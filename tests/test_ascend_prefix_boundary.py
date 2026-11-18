"""End-to-end checks for the single-request donor-boundary experiment."""

import copy

import pytest
import torch

from benchmarks.ascend_prefix_boundary import boundary_experiment
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model


def previous_pass(model, case):
    return {
        "runs": {
            mode: replay_mode(model, "torch", case, mode, output_tokens=3, seed=168)
            for mode in ("off", "split", "reuse")
        }
    }


@pytest.mark.parametrize("suffix", [8, 24])
def test_donor_boundary_controls_observe_the_same_target(suffix):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case(f"p32_s{suffix}", "boundary", 32, suffix)
    previous = previous_pass(model, case)
    wrong_input = copy.deepcopy(previous)
    wrong_input["runs"]["reuse"]["requests"][0]["prompt_sha256"] = "0" * 64
    wrong_start = copy.deepcopy(previous)
    wrong_start["runs"]["reuse"]["requests"][0]["prefill_start"] = 0
    result = boundary_experiment(model, "torch", case, 3, 168, [wrong_input, wrong_start, previous])
    summary, modes = result["summary"], result["modes"]
    assert summary["matched_prior_replays"] == [3]
    assert summary["comparable"] and not summary["control_issues"]

    full = [[0, 32 + suffix]]
    chunked = [[0, 32], [32, suffix]]
    tail = [[32, suffix]]
    expected = {
        "off": (full, full, 0),
        "split": (chunked, chunked, 0),
        "reuse": (full, tail, 32),
        "aligned": (chunked, tail, 32),
    }
    digest = previous["runs"]["off"]["requests"][0]["prompt_sha256"]
    for mode, (donor, target, hits) in expected.items():
        run = modes[mode]
        assert run["request"]["prompt_sha256"] == digest
        assert run["donor_spans"] == donor and run["target_spans"] == target
        assert run["prefix_queries"] == 1 and run["prefix_hits"] == hits
        assert [logit["selected_id"] for logit in run["logits"]] == run["request"]["tokens"]

    aligned = summary["kv"]["aligned_donor__split_target"]
    assert aligned["equal"]
    assert (
        aligned["compared_components"]
        == 2 * model.config.total_ut_steps * model.config.num_hidden_layers
    )
    assert summary["output_first_diff"]["split__aligned"] is None


def test_boundary_requires_one_jointly_matching_prior_pass():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("p32_s8", "boundary", 32, 8)
    wrong_digest = previous_pass(model, case)
    wrong_output = copy.deepcopy(wrong_digest)
    wrong_digest["runs"]["split"]["requests"][0]["prompt_sha256"] = "0" * 64
    wrong_output["runs"]["reuse"]["requests"][0]["tokens"][0] ^= 1
    result = boundary_experiment(model, "torch", case, 3, 168, [wrong_digest, wrong_output])
    summary = result["summary"]
    assert not summary["control_issues"]
    assert summary["matched_prior_replays"] == []
    assert not summary["comparable"]
