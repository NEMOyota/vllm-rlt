"""End-to-end controls for testing tiled Ouro projections against the original cold path."""

import copy
import hashlib
from contextlib import contextmanager

import pytest
import torch

from benchmarks import ascend_prefix_fix_trial as experiment
from benchmarks.ascend_prefix_head import HeadProbe
from benchmarks.ascend_prefix_logits_probe import LogitsProbe
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_config, tiny_ouro_model
from vllm_rlt.models import OuroForCausalLM


@pytest.fixture(scope="module")
def archived_case():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("short", "fix", 32, 8)
    prior = {
        "runs": {arm: replay_mode(model, "torch", case, arm, 3, 168) for arm in ("off", "reuse")}
    }
    stale = copy.deepcopy(prior)
    for run in stale["runs"].values():
        run["requests"][0]["prompt_sha256"] = "0" * 64
    return model, case, prior, stale


CONTRACT = {"output_tokens": 3, "seed": 168}


def test_fix_trial_reproduces_joint_native_controls_and_observes_tiling(archived_case):
    model, case, prior, stale = archived_case
    rows = 16
    inconsistent = copy.deepcopy(prior)
    inconsistent["runs"]["off"]["requests"][0]["prompt_sha256"] = "0" * 64
    result = experiment.fix_trial(
        model, "torch", case, CONTRACT, [inconsistent, stale, prior], rows
    )
    assert result["matched_prior_replays"] == [3]
    assert result["comparable"] and not result["controls"]
    assert result["hit_tokens"] == {"off": [0], "reuse": [32]}
    assert result["native_off_vs_reuse"] == []
    assert not result["head"]
    producer = result["layer0_different_elements"]
    assert set(producer) == {"native", "candidate"}
    assert set(producer["native"]) == {"k_proj", "v_proj"}
    assert producer["candidate"] == {"k_proj": 0, "v_proj": 0}
    for arm in ("off", "reuse"):
        work = result["projection_work"][arm]
        assert work["tiles"] >= work["calls"] > 0
        assert work["padded_rows"] > 0
        assert work["input_rows"] + work["padded_rows"] == rows * work["tiles"]
    comparisons = (
        "candidate_off_vs_reuse",
        "native_off_vs_candidate_off",
        "native_off_vs_candidate_reuse",
    )
    assert result["preserves_baseline_and_parity"] == all(not result[key] for key in comparisons)

    incompatible = experiment.fix_trial(model, "torch", case, CONTRACT, [stale], rows)
    assert incompatible["matched_prior_replays"] == []
    assert not incompatible["comparable"] and not incompatible["preserves_baseline_and_parity"]


def test_candidate_self_parity_cannot_mask_original_cold_output_drift(archived_case, monkeypatch):
    model, case, prior, _ = archived_case
    original = prior["runs"]["off"]["requests"][0]["tokens"][0]
    forced = (original + 1) % model.config.vocab_size
    tile = experiment.fixed_kv_rows

    def change_winner(_, inputs, logits):
        changed = logits.clone()
        changed[..., forced] = torch.finfo(logits.dtype).max
        return changed

    @contextmanager
    def drifting_candidate(model, rows):
        with tile(model, rows) as counters, model.lm_head.register_forward_hook(change_winner):
            yield counters

    monkeypatch.setattr(experiment, "fixed_kv_rows", drifting_candidate)
    result = experiment.fix_trial(model, "torch", case, CONTRACT, [prior], 16)
    drift = [{"request_id": "target-0", "position": 1, "ids": [original, forced]}]
    assert result["comparable"] and result["candidate_off_vs_reuse"] == []
    assert result["native_off_vs_candidate_off"] == result["native_off_vs_candidate_reuse"] == drift
    assert not result["preserves_baseline_and_parity"]


def test_head_probe_tracks_target_and_position_in_a_real_c4_coda():
    # This seed produces distinct top logits across both batch rows and output positions.
    torch.manual_seed(11)
    model = OuroForCausalLM(tiny_ouro_config()).to(dtype=torch.bfloat16)
    case = Case("c4_h4", "fix", 32, 16, concurrency=4, hits=4)
    captures = {}
    rid, position = "target-2", 2

    def logits_only(engine, targets, donor):
        captures["logits"] = LogitsProbe(engine.model_runner)
        return captures["logits"]

    replay_mode(model, "torch", case, "reuse", 3, 168, target_context=logits_only)
    logits = captures["logits"].report()
    decision = logits[rid][position - 1]
    assert decision["top_logits"] != logits["target-0"][position - 1]["top_logits"]
    assert decision["top_logits"] != logits[rid][0]["top_logits"]
    ids = list(reversed(decision["top_ids"]))

    def head_only(engine, targets, donor):
        captures["head"] = HeadProbe(engine.model_runner, model, rid, position, ids)
        return captures["head"]

    run = replay_mode(model, "torch", case, "reuse", 3, 168, target_context=head_only)
    probe = captures["head"]
    report = probe.report()
    assert (report["request_id"], report["position"], report["ids"]) == (rid, position, ids)
    assert report["selected_id"] == run["requests"][2]["tokens"][1] == decision["selected_id"]
    assert (
        report["head_logits"] == report["sampler_logits"] == list(reversed(decision["top_logits"]))
    )
    assert (
        report["hidden_sha256"]
        == hashlib.sha256(probe.hidden.view(torch.uint8).numpy()).hexdigest()
    )
    fp64 = model.lm_head.weight[ids].double() @ probe.hidden.double()
    assert report["fp64_logits"] == pytest.approx(fp64.detach().tolist(), abs=1e-12)

    off = replay_mode(model, "torch", case, "off", 3, 168)
    experiment_result = experiment.fix_trial(
        model, "torch", case, CONTRACT, [{"runs": {"off": off, "reuse": run}}], 16
    )
    assert experiment_result["comparable"] and not experiment_result["controls"]
    assert experiment_result["layer0_different_elements"] is None
    assert experiment_result["hit_tokens"]["reuse"] == [32] * 4
