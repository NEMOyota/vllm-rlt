"""Localize a C4 value difference to the packed projection, KV write, or logical readback."""

import copy
from unittest.mock import patch

import pytest
import torch

from benchmarks.ascend_prefix_c4_slot import cold_slot_run
from benchmarks.ascend_prefix_c4_writer import writer_experiment
from benchmarks.ascend_prefix_c4_writer_probe import FirstWriteProbe
from benchmarks.ascend_prefix_workload import Case
from tests.helpers import tiny_ouro_model
from vllm_rlt.core.kv_cache_manager import KVCacheManager

CASE = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
CONTRACT = {"seed": 168, "output_tokens": 3}
IDS = [f"target-{i}" for i in range(4)]


def replay_and_observe(model, variant="clones"):
    prior = cold_slot_run(model, "torch", CASE, CONTRACT, variant)
    result = writer_experiment(model, "torch", CASE, CONTRACT, prior, variant=variant)
    return prior, result


@pytest.mark.parametrize("variant", ["clones", "base"])
def test_writer_uses_matching_cold_archive_and_observes_all_four_slots(variant):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    prior, result = replay_and_observe(model, variant)
    summary = result["summary"]
    assert summary["comparable"] and not summary["controls"]
    assert not summary["address_issue"]
    assert summary["write_read_mismatch"] == {rid: {"k": 0, "v": 0} for rid in IDS}
    stages = result["stages"]
    assert set(stages) == {
        "hidden",
        "norm",
        "v_input",
        "v_proj",
        "k_write",
        "v_write",
        "k_read",
        "v_read",
    }
    for stats in stages.values():
        assert set(stats) == set(IDS)
        assert all(s["elements"] > 0 for s in stats.values())
    run = result["run"]
    assert run["variant"] == variant
    assert run["prefix_queries"] == 4 and run["prefix_hits"] == 0
    assert [row["request_id"] for row in run["requests"]] == IDS
    assert run["requests"] == prior["requests"]
    if variant == "clones":
        assert len({row["prompt_sha256"] for row in run["requests"]}) == 1
        assert summary["first_v_difference"] == dict.fromkeys(IDS)
        tampered = copy.deepcopy(prior)
        tampered["prefill_fingerprints"]["target-0"]["sha256"][0][0]["v"] = "0" * 64
        invalid = writer_experiment(model, "torch", CASE, CONTRACT, tampered)["summary"]
        assert not invalid["comparable"] and invalid["controls"]


def test_writer_identifies_a_slot_specific_v_projection_change_before_storage():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    packed_rows = CASE.concurrency * (CASE.prefix + CASE.suffix)

    def inject_slot_difference(_, inputs, output):
        if output.shape[0] != packed_rows:
            return output
        output = output.clone()
        output[0, 0] += 1
        return output

    with model.model.layers[0].self_attn.v_proj.register_forward_hook(inject_slot_difference):
        _, result = replay_and_observe(model)
    summary = result["summary"]
    assert summary["comparable"] and not summary["address_issue"]
    assert summary["first_v_difference"]["target-0"] == "v_proj"
    for name in ("hidden", "norm", "v_input"):
        assert result["stages"][name]["target-0"]["different_elements"] == 0
    for name in ("v_proj", "v_write", "v_read"):
        assert result["stages"][name]["target-0"]["different_elements"] == 1
    sample = result["stages"]["v_proj"]["target-0"]["sample"]
    assert (sample["token_0based"], sample["channel"]) == (0, 0)
    assert abs(sample["focus"] - sample["fp64_dot"]) < abs(sample["slot"] - sample["fp64_dot"])
    assert not summary["connection_mismatches"]
    assert summary["write_read_mismatch"] == {rid: {"k": 0, "v": 0} for rid in IDS}


def test_writer_detects_a_changed_readback_with_identical_projection_and_write_inputs():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    original = KVCacheManager._write_prepared
    packed_rows = CASE.concurrency * (CASE.prefix + CASE.suffix)

    def corrupt_after_write(cache, layer, batch, k, v):
        original(cache, layer, batch, k, v)
        if layer == 0 and len(batch.rows) == packed_rows and batch.rows[0][1] == 0:
            block = int(batch.write_blocks[0])
            offset = int(batch.write_offsets[0])
            cache.value_cache[block, layer, offset, 0, 0] += 1

    with patch.object(KVCacheManager, "_write_prepared", corrupt_after_write):
        _, result = replay_and_observe(model)
    summary = result["summary"]
    assert not summary["address_issue"]
    assert summary["first_v_difference"]["target-0"] == "v_read"
    assert summary["write_read_mismatch"]["target-0"] == {"k": 0, "v": 1}
    for rid in IDS[1:]:
        assert summary["write_read_mismatch"][rid] == {"k": 0, "v": 0}


def test_writer_checks_scatter_indices_even_if_staged_addresses_match():
    cache = KVCacheManager(2, 1, 4, 32, 2, 4, dtype=torch.bfloat16)
    probe = FirstWriteProbe(tiny_ouro_model(dtype=torch.bfloat16), cache, IDS, 2, 4)
    rows = []
    for rid in IDS:
        assert cache.allocate(rid, 4)
        table = cache.get_block_table(rid, 0)
        rows.extend((table[pos // 2], 0, pos % 2) for pos in range(4))
    probe._check_addresses(rows, list(range(4)) * 4)
    assert probe.address_issue is None
    block, _, offset = rows[8]
    rows[8] = (block, 1, offset)
    probe._check_addresses(rows)
    assert probe.address_issue["kind"] == "scatter_index"
    assert probe.address_issue["request_id"] == "target-2"
    assert probe.address_issue["position"] == 0
