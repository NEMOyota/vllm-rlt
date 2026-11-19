"""Recreate the layer-0 projection operands of the C1 donor and C4 cloned focus target."""

import hashlib

import torch

from benchmarks.ascend_prefix_c4_slot import _digest
from benchmarks.ascend_prefix_workload import _prompts


@torch.inference_mode()
def prepare_linear_inputs(model, case, contract):
    """Return the device activation, original projection weights and input provenance."""
    is_c4 = case.name == "c4_h4"
    if case.name not in ("total512_p384", "c4_h4") or case.concurrency != (4 if is_c4 else 1):
        raise ValueError("expected the single-request donor or the four-request clone experiment")

    prefix = case.reusable_tokens
    donor, targets, _ = _prompts(
        case,
        "off" if is_c4 else "reuse",
        contract["output_tokens"],
        model.config.vocab_size,
        contract["seed"],
    )
    prompt = targets[2] if is_c4 else donor
    if not 0 < prefix < len(prompt):
        raise ValueError("expected a full-page prefix followed by a nonempty suffix")
    layer = model.model.layers[0]
    device = model.model.embed_tokens.weight.device

    def normalized(token_ids):
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        hidden = layer.input_layernorm(model.prelude(tokens))
        if hidden.dtype != torch.bfloat16 or hidden.shape != (
            len(tokens),
            model.config.hidden_size,
        ):
            raise ValueError("expected packed BF16 activations at the first layer")
        return hidden

    # The C4 writer arranges all 144 tokens of each clone contiguously, then
    # the next clone. Run that packed normalization once before the isolated ops.
    packed = normalized(prompt * case.concurrency)
    x = packed.contiguous()
    single = normalized(prompt if is_c4 else prompt[:prefix])
    host = x.detach().cpu()
    reference = single.detach().cpu().contiguous()
    groups = host.split(len(prompt)) if is_c4 else (host[:prefix],)
    differences = [
        int((group.view(torch.int16) != reference.view(torch.int16)).sum()) for group in groups
    ]
    controls = []
    if any(differences):
        controls.append("packed_slot_vs_single_input" if is_c4 else "full_vs_chunk_input")
    if not torch.isfinite(host).all().item() or not torch.isfinite(reference).all().item():
        controls.append("nonfinite_projection_input")

    weights = {}
    for name in ("k_proj", "v_proj"):
        projection = getattr(layer.self_attn, name)
        weight = projection.weight
        if (
            projection.bias is not None
            or weight.dtype != torch.bfloat16
            or weight.device != device
            or weight.ndim != 2
            or weight.shape[1] != x.shape[1]
        ):
            raise ValueError(f"{name} requires an unmodified bias-free BF16 linear projection")
        weights[name] = weight
    info = dict(
        case=case.name,
        depth=0,
        layer=0,
        source_request="target-2" if is_c4 else "donor",
        focus_slot=2 if is_c4 else None,
        prompt_sha256=_digest(prompt),
        prefix_sha256=_digest(prompt[:prefix]),
        input_sha256=hashlib.sha256(host.view(torch.uint8).numpy()).hexdigest(),
        prefix_rows=prefix,
        slot_rows=len(prompt) if is_c4 else None,
        input_shape=list(x.shape),
        norm_output_stride=list(packed.stride()),
        norm_output_storage_offset=packed.storage_offset(),
        linear_input_stride=list(x.stride()),
        input_control_different_elements=differences,
        controls=controls,
    )
    return x, weights, info
