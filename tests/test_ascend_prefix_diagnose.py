"""Exercise the prefix diagnosis with real CPU engine, scheduler and sampler outputs."""

import copy
from contextlib import contextmanager

import pytest
import torch

from benchmarks.ascend_prefix_diagnose import diagnose_pass
from benchmarks.ascend_prefix_evidence import archive_evidence, compare_replays
from benchmarks.ascend_prefix_kv_probe import KVProbe
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_scenarios import summarize
from benchmarks.ascend_prefix_workload import Case, run_trial
from tests.helpers import tiny_ouro_config, tiny_ouro_model
from vllm_rlt.models import OuroForCausalLM


@pytest.fixture(scope="module")
def diagnostic():
    # This seed yields different greedy decisions across the packed target rows.
    torch.manual_seed(11)
    model = OuroForCausalLM(tiny_ouro_config()).to(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    order = ("off", "miss", "reuse", "split")
    trial = {arm: run_trial(model, "torch", case, arm, 3, 168) for arm in order[:3]}
    entry = {"summary": summarize(case, [trial]), "warmups": [], "trials": [trial]}
    result = diagnose_pass(model, "torch", case, entry, 3, 168, order)
    return entry, result


def test_replay_records_real_prefill_spans_and_sampler_decisions(diagnostic):
    entry, result = diagnostic
    (original,) = entry["trials"]
    runs = result["runs"]
    assert set(runs) == {"off", "miss", "reuse", "split"}
    assert len({tuple(row["tokens"]) for row in runs["off"]["requests"]}) > 1
    for arm, run in runs.items():
        expected_start = 32 if arm == "reuse" else 0
        assert run["prefix_hits"] == 4 * expected_start
        assert run["prefix_queries"] == 4
        assert run["kv"]["ok"], run["kv"]
        for target, row in enumerate(run["requests"]):
            request_id = f"target-{target}"
            assert row["request_id"] == request_id
            assert row["hit_tokens"] == row["prefill_start"] == expected_start
            baseline = original["off" if arm == "split" else arm]["requests"][target]
            assert row["prompt_sha256"] == baseline["prompt_sha256"]
            assert row["tokens"] == baseline["tokens"]
            decisions = run["logits"][request_id]
            assert [d["position"] for d in decisions] == [1, 2, 3]
            assert [d["selected_id"] for d in decisions] == row["tokens"]
            for selected, decision in zip(row["tokens"], decisions, strict=True):
                winner, other = decision["top_ids"]
                assert winner == selected and other != selected
                top1, top2 = decision["top_logits"]
                assert top1 >= top2
                assert decision["margin"] == pytest.approx(top1 - top2)

    def prefill_spans(arm):
        return {
            (span["token_start"], span["token_count"])
            for step in runs[arm]["steps"]
            for span in step["spans"]
            if span["request_id"] == "target-0" and span["phase"] == "prefill"
        }

    assert prefill_spans("off") == {(0, 48)}
    assert prefill_spans("miss") == {(0, 48)}
    assert prefill_spans("reuse") == {(32, 16)}
    assert prefill_spans("split") == {(0, 32), (32, 16)}
    checks = result["comparison"]
    assert checks["status"] == "PASS", checks["primary_issues"]
    assert checks["joint_archive_outputs_on_same_inputs_1based"]["trials"] == [1]
    assert checks["arm_checks"]["reuse"]["checked_kv_pages"] == 8


def test_comparison_locates_divergent_target_token_and_original_logits(diagnostic):
    entry, result = diagnostic
    runs = copy.deepcopy(result["runs"])
    # Target-2 diverges earlier than target-0, despite its later batch row.
    for target, offset in ((0, 2), (2, 1)):
        decision = runs["reuse"]["logits"][f"target-{target}"][offset]
        runs["reuse"]["requests"][target]["tokens"][offset] = decision["top_ids"][1]
    comparison = compare_replays(runs, entry)
    assert comparison["status"] == "FAIL"
    assert "off_reuse:tokens" in comparison["primary_issues"]
    difference = comparison["pairs"]["off_reuse"]
    assert difference["changed_targets"] == [0, 2]
    first = difference["first_diff"]
    assert (first["target"], first["position_1based"]) == (2, 2)
    for arm in ("off", "reuse"):
        assert first["token_ids"][arm] == runs[arm]["requests"][2]["tokens"][1]
        assert first["top2"][arm] == runs[arm]["logits"]["target-2"][1]
    assert comparison["joint_archive_outputs_on_same_inputs_1based"]["trials"] == []


def test_archive_can_locate_a_warmup_failure_after_a_clean_replay(diagnostic):
    original_entry, result = diagnostic
    entry = copy.deepcopy(original_entry)
    warmup = copy.deepcopy(entry["trials"][0])
    warmup["reuse"]["requests"][3]["tokens"][1] ^= 1
    entry["warmups"].append(warmup)
    archive = archive_evidence(entry, Case(**entry["summary"]["case"]))
    assert archive["status"] == "FAIL"
    warm = archive["phases"]["warmups"]
    assert warm["failed_trials_1based"] == [1]
    assert warm["examples"][0]["target"] == 3
    assert warm["examples"][0]["first_diff_1based"] == 2
    assert archive["phases"]["trials"]["failed_trials_1based"] == []
    replay = compare_replays(result["runs"], entry)
    assert replay["status"] == "PASS"
    assert replay["joint_archive_outputs_on_same_inputs_1based"] == {"warmups": [], "trials": [1]}


def test_shared_value_page_byte_flip_survives_to_diagnostic_report():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("hot", "replay", 32, 16)
    observed = {}

    @contextmanager
    def corrupt_at_end(engine, targets, donor):
        cache = engine.cache_manager
        probe = KVProbe(cache, [32])
        with probe:
            yield
            page = probe.donor_prefix_pages[0][0]
            # Flip just one bit in V at one physical depth/page. All other
            # donor prefix tensors retain their original bytes.
            cache.value_cache[page].view(torch.uint8).reshape(-1)[0] ^= 1
        observed.update(page=page, report=probe.report())

    replay_mode(model, "torch", case, "reuse", 2, 168, engine_context=corrupt_at_end)
    report = observed["report"]
    assert not report["ok"]
    (change,) = report["changed_pages"]
    assert (change["kind"], change["depth"], change["logical_page"]) == ("value", 0, 0)
    assert change["physical_page"] == observed["page"]
    assert change["before_sha256"] != change["after_sha256"]
