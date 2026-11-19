"""Cold-slot and frozen-operand checks for the C4 prefix consumer experiment."""

import copy
import hashlib
import struct
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from benchmarks.ascend_prefix_attention_replay import FrozenAttention
from benchmarks.ascend_prefix_c4_consumer import consumer_experiment
from benchmarks.ascend_prefix_logical_kv import LogicalKVProbe
from benchmarks.ascend_prefix_replay import replay_mode
from benchmarks.ascend_prefix_workload import Case, _prompts
from tests.helpers import tiny_ouro_model
from vllm_rlt.kernels.ascend_eager import AscendEagerAttention


def _digest(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}q", *tokens)).hexdigest()


def test_consumer_holds_prefix_and_schedule_fixed_and_checks_original_cold_run():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    contract = {"seed": 168, "output_tokens": 3}
    ids = [f"target-{i}" for i in range(4)]
    probes = {}

    def capture_off(engine, targets, donor):
        probes["kv"] = LogicalKVProbe(engine.cache_manager, case.prefix, ids)
        return probes["kv"]

    old = replay_mode(model, "torch", case, "off", 3, 168, engine_context=capture_off)
    old["fingerprints"] = probes["kv"].report()
    out = consumer_experiment(model, "torch", case, contract, old)
    summary, runs = out["summary"], out["runs"]
    assert summary["comparable"] and not summary["controls"]
    assert summary["frozen_attention"] is None
    original = _prompts(case, "off", 3, model.config.vocab_size, 168)[1]
    future = copy.deepcopy(original)
    future[2][case.prefix] = (future[2][case.prefix] + 1) % model.config.vocab_size
    expected = {"base": original, "repeat": original, "future": future, "clones": [original[2]] * 4}
    prefix_digest = _digest(original[2][: case.prefix])
    old_schedule = [step["spans"] for step in old["steps"]]
    for name, prompts in expected.items():
        run = runs[name]
        assert [row["request_id"] for row in run["requests"]] == ids
        assert [row["prompt_sha256"] for row in run["requests"]] == [_digest(t) for t in prompts]
        assert run["prefix_sha256"] == dict.fromkeys(ids, prefix_digest)
        assert [step["spans"] for step in run["steps"]] == old_schedule
        assert run["prefix_hits"] == 0 and run["prefix_queries"] == 4
        assert all(run["prefill_fingerprints"][rid]["tap"] == "after_prefill" for rid in ids)
        assert all(run["fingerprints"][rid]["tap"] == "before_free" for rid in ids)
    assert all(change["equal"] for change in summary["future_prefix_changes"].values())
    assert all(change["equal"] for change in summary["clone_slot_vs_focus"].values())
    assert all(not changed for changed in summary["changed_between_prefill_and_release"].values())

    stale = copy.deepcopy(old)
    stale["requests"][2]["prompt_sha256"] = "0" * 64
    rejected = consumer_experiment(model, "torch", case, contract, stale)["summary"]
    assert not rejected["comparable"] and "base:archived_cold_run" in rejected["controls"]


def test_frozen_attention_replays_the_real_eager_suffix_and_keeps_original_outputs():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    case = Case("c4_h4", "concurrency", 32, 16, concurrency=4, hits=4)
    ids = [f"target-{i}" for i in range(4)]
    endpoint = case.prefix + case.suffix

    def replay(observe, *, alter_shadow=False):
        returned = []
        probes = {}

        @contextmanager
        def inspect(engine, targets, donor):
            cache = engine.cache_manager
            with ExitStack() as hooks:
                # Use real BF16 Ascend eager operations on CPU with the ordinary CPU KV writer.
                kernel = AscendEagerAttention()
                hooks.enter_context(patch.object(cache.attention, "kernel", kernel))
                if observe:
                    probes["frozen"] = hooks.enter_context(
                        FrozenAttention(cache, ids, case.prefix, endpoint)
                    )
                    if alter_shadow:
                        shadow = probes["frozen"].shadow
                        continuation = shadow.attend_metadata

                        def changed_continuation(*args):
                            suffix = continuation(*args).clone()
                            suffix[0, 0, 0] += 1
                            return suffix

                        hooks.enter_context(
                            patch.object(shadow, "attend_metadata", changed_continuation)
                        )
                original = cache._attend_prepared

                def save_output(layer, batch, q):
                    output = original(layer, batch, q)
                    returned.append(output.detach().clone())
                    return output

                hooks.enter_context(patch.object(cache, "_attend_prepared", save_output))
                yield

        run = replay_mode(model, "torch", case, "off", 2, 168, target_context=inspect)
        return run, returned, probes

    reference, expected, _ = replay(False)
    observed, actual, probes = replay(True)
    checked = probes["frozen"].report()
    count = model.config.total_ut_steps * model.config.num_hidden_layers
    assert checked["checked_calls"] == checked["expected_calls"] == count
    assert not checked["control_issues"]
    assert checked["first_gather_mismatch"] is None
    assert set(checked["per_request"]) == set(ids)
    assert len(actual) == len(expected) >= count
    for before, after in zip(expected, actual, strict=True):
        assert torch.equal(before.view(torch.int16), after.view(torch.int16))
    assert observed["requests"] == reference["requests"]

    # Even a demonstrably different shadow tensor must stay out of model execution.
    fault, returned, probes = replay(True, alter_shadow=True)
    diff = probes["frozen"].report()["first_output_diff"]
    assert (diff["depth"], diff["layer"], diff["request_id"]) == (0, 0, "target-0")
    for before, after in zip(expected, returned, strict=True):
        assert torch.equal(before.view(torch.int16), after.view(torch.int16))
    assert fault["requests"] == reference["requests"]
