from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from benchmarks.ascend_prefix_projection import fixed_kv_rows
from tests.helpers import tiny_ouro_model


@torch.inference_mode()
def test_fixed_kv_rows_uses_original_bf16_linears_and_zero_pads_only_the_tail():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    projections = [
        proj
        for layer in model.model.layers
        for proj in (layer.self_attn.k_proj, layer.self_attn.v_proj)
    ]
    with ExitStack() as stack:
        originals = [
            stack.enter_context(patch.object(p, "forward", wraps=p.forward)) for p in projections
        ]
        with fixed_kv_rows(model, 4) as work:
            for layer in model.model.layers:
                assert all(
                    "forward" not in vars(p)
                    for p in (layer.self_attn.q_proj, layer.self_attn.o_proj)
                )
            for proj in projections:
                for n in (1, 4, 6):
                    x = torch.zeros(proj.in_features, n, dtype=torch.bfloat16).T
                    x[:, 0] = torch.arange(1, n + 1)
                    expected = (x[:, :1].float() * proj.weight[:, 0].float()).bfloat16()
                    assert torch.equal(proj(input=x), expected)
        for proj, original in zip(projections, originals, strict=True):
            assert proj.forward is original
            inputs = [call.args[0] for call in original.call_args_list]
            assert len(inputs) == 4
            assert all(x.shape == (4, proj.in_features) and x.is_contiguous() for x in inputs)
            assert torch.equal(inputs[0][1:], torch.zeros_like(inputs[0][1:]))
            assert torch.equal(inputs[-1][2:], torch.zeros_like(inputs[-1][2:]))
        nproj = len(projections)
        assert work == dict(
            calls=3 * nproj, input_rows=11 * nproj, tiles=4 * nproj, padded_rows=5 * nproj
        )
    assert all("forward" not in vars(p) for p in projections)


def test_fixed_kv_rows_restores_all_hooks_on_invalid_call_and_partial_setup():
    model = tiny_ouro_model(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="packed 2D BF16"):
        with fixed_kv_rows(model, 8):
            model.model.layers[0].self_attn.k_proj(torch.ones(2, model.config.hidden_size))
    model.model.layers[-1].self_attn.v_proj.float()
    with pytest.raises(ValueError, match="BF16 linear weights"):
        with fixed_kv_rows(model, 8):
            pytest.fail("a mixed-dtype model must fail during setup")
    for layer in model.model.layers:
        assert all(
            "forward" not in vars(p) for p in (layer.self_attn.k_proj, layer.self_attn.v_proj)
        )
