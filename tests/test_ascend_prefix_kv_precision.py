import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from benchmarks import ascend_prefix_kv_precision as bench
from benchmarks.ascend_prefix_projection import _without_hf32, fp32_kv_projections
from benchmarks.ascend_prefix_scenarios import ARMS, suites
from benchmarks.ascend_prefix_workload import expected_hits
from tests.helpers import tiny_ouro_model


@pytest.mark.parametrize("projection", ["k", "v", "kv"])
@pytest.mark.parametrize("rows", [None, 4])
@torch.inference_mode()
def test_only_selected_linears_use_fp32_and_return_bf16(projection, rows):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    selected = {f"model.layers.1.self_attn.{name}_proj" for name in projection}
    modules = dict(model.named_modules())
    hidden = torch.zeros(model.config.hidden_size, 6, dtype=torch.bfloat16).T
    hidden[:, 0] = torch.arange(1, 7)
    linear = F.linear
    calls = []

    def observe(x, weight, bias):
        assert not torch.is_autocast_enabled("cpu")
        assert x.dtype == weight.dtype == torch.float32
        calls.append((x.clone(), weight))
        return linear(x, weight, bias)

    with fp32_kv_projections(model, projection, [1], rows) as work:
        assert {name for name, m in modules.items() if "forward" in vars(m)} == selected
        for name in selected:
            module = modules[name]
            calls.clear()
            with patch.object(F, "linear", observe), torch.autocast("cpu", dtype=torch.bfloat16):
                actual = module(input=hidden)
            assert module.weight.dtype == actual.dtype == torch.bfloat16
            expected = linear(hidden.float(), module.weight.float()).bfloat16()
            assert torch.equal(actual, expected)
            assert [len(x) for x, _ in calls] == ([6] if rows is None else [4, 4])
            assert all(weight is calls[0][1] for _, weight in calls)
            inputs = torch.cat([x for x, _ in calls])
            assert torch.equal(inputs[:6], hidden.float())
            assert torch.count_nonzero(inputs[6:]) == 0
        assert len(work) == len(selected)
        for counts in work.values():
            assert (counts["calls"], counts["input_rows"], counts["padded_rows"]) == (
                1,
                6,
                0 if rows is None else 2,
            )
    assert all("forward" not in vars(m) for m in modules.values())


def test_projection_restores_on_error_and_rejects_an_empty_selection():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="packed 2D BF16"), fp32_kv_projections(model):
        model.model.layers[0].self_attn.v_proj(torch.ones(2, model.config.hidden_size))
    assert all("forward" not in vars(m) for m in model.modules())
    for layers in ([], [-1], [len(model.model.layers)], [0, 0]):
        with pytest.raises(ValueError), fp32_kv_projections(model, layers=layers):
            pytest.fail("invalid layer selection accepted")

    flags = SimpleNamespace(allow_hf32=True)

    def fail(_):
        assert not flags.allow_hf32
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _without_hf32(flags, fail, None)
    assert flags.allow_hf32


def test_report_separates_cache_parity_from_changed_cold_outputs(monkeypatch):
    model = tiny_ouro_model(dtype=torch.bfloat16)
    presets = {c.name: c for c in suites()["full"]}
    cases = [presets[name] for name in bench.DEFAULT_CASES]
    args = SimpleNamespace(
        repeats=2, output_tokens=2, seed=168, projection="v", layers=[1], chunk_rows=4
    )

    def simulate(model, backend, case, arm, output_tokens, seed):
        layer = model.model.layers[1].self_attn.v_proj
        patched = "forward" in vars(layer)
        layer(torch.zeros(1, layer.in_features, dtype=torch.bfloat16))
        _, hits = expected_hits(case, arm)
        requests = []
        for i, hit in enumerate(hits):
            second = 8
            if case == cases[0] and arm == "reuse" and not patched:
                second = 11
            if case == cases[1] and arm == "reuse" and patched:
                second = 12
            if case == cases[2] and patched and i == 2:
                second = 13
            requests.append(
                dict(
                    prompt_sha256=f"p{i}",
                    tokens=[7, second],
                    hit_tokens=hit,
                    prefill_start=hit,
                    expected_hit_tokens=hit,
                )
            )
        return dict(requests=requests, prefix_hits=sum(hits), prefix_queries=len(hits))

    monkeypatch.setattr(bench, "run_trial", simulate)
    stream = io.StringIO()
    review = bench.run_cases(model, "torch", cases, args, stream)
    assert review["cache_parity_failed"] == [cases[1].name]
    assert review["native_cold_changed"] == [cases[2].name]
    assert review["invalid"] == []
    *trials, last = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(trials) == 6 and last["kind"] == "review"
    assert trials[0]["passed"] and trials[0]["native_off_vs_reuse"]
    assert trials[2]["kv_fp32_parity_diffs"][0]["first_diff_1based"] == 2
    assert trials[4]["cache_parity"] == "PASS" and not trials[4]["passed"]
    assert trials[4]["native_off_vs_kv_fp32_off"] == [[2, 2, 8, 13]]
    assert trials[4]["kv_fp32_reuse_hits"] == [128] * 4
    for arm in ARMS:
        assert trials[0]["projection_work"][arm]["layers.1.v_proj"]["calls"] == 1
