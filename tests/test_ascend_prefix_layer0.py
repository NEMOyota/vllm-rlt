"""Verify the first-layer producer against signatures captured from real engine KV pages."""

import copy

import pytest
import torch

from benchmarks.ascend_prefix_boundary import boundary_experiment
from benchmarks.ascend_prefix_layer0 import _compare, layer0_experiment
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model


@pytest.fixture(scope="module")
def boundary_record():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("p32_s8", "prefix_length", 32, 8)
    runs = {
        mode: replay_mode(model, "torch", case, mode, 3, 168) for mode in ("off", "split", "reuse")
    }
    boundary = boundary_experiment(model, "torch", case, 3, 168, [{"runs": runs}])
    assert boundary["summary"]["comparable"]
    boundary["contract"] = {"seed": 168, "output_tokens": 3}
    return model, case, boundary


def test_layer0_producer_matches_both_engine_snapshots(boundary_record):
    model, case, boundary = boundary_record
    result = layer0_experiment(model, case, boundary)
    assert result["matched_boundary"], result
    stages = result["stages"]
    hidden_elements = case.reusable_tokens * model.config.hidden_size
    kv_elements = case.reusable_tokens * model.config.num_key_value_heads * model.config.head_dim
    assert stages["embedding"]["elements"] == stages["norm"]["elements"] == hidden_elements
    assert stages["embedding"]["different_elements"] == 0
    for name in ("k_proj", "v_proj", "k_write", "v_write"):
        assert stages[name]["elements"] == kv_elements
        assert 0 <= stages[name]["different_elements"] <= kv_elements
    for name in ("rope_cos", "rope_sin"):
        assert stages[name]["elements"] == case.reusable_tokens * model.config.head_dim


def test_layer0_rejects_one_inconsistent_archived_component(boundary_record):
    model, case, original = boundary_record
    tampered = copy.deepcopy(original)
    tampered["modes"]["aligned"]["fingerprint"]["sha256"][0][0]["k"] = "0" * 64
    result = layer0_experiment(model, case, tampered)
    assert not result["matched_boundary"]
    assert result["matched_cache_kv"] == {
        "full": {"k": True, "v": True},
        "chunk": {"k": False, "v": True},
    }


def test_layer0_counts_bf16_bit_differences_separately_from_absolute_error():
    left = torch.tensor([[0.0, 3.0], [1.0, 2.0]], dtype=torch.bfloat16).T
    right = left.clone()
    right[0, 0] = -0.0
    right[0, 1] = 1.5
    delta = _compare(left, right)
    assert delta["elements"] == 4
    assert delta["different_elements"] == 2
    assert delta["max_abs"] == pytest.approx(0.5)
    assert delta["worst_index"] == 1
