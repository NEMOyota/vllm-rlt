"""Locate shape-dependent differences before the first Ouro KV write of a C1 replay."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.ascend_prefix_scenarios import suites
from benchmarks.ascend_prefix_workload import _prompts
from benchmarks.preemption import checkpoint_identity
from vllm_rlt.device import resolve_device
from vllm_rlt.layers.rotary_embedding import apply_rotary_pos_emb
from vllm_rlt.models import OuroForCausalLM


@torch.inference_mode()
def _prefix_stages(model, prompt, prefix, *, names=None):
    # Mirror OuroForCausalLM.prelude and the first layer up to _write_prepared.
    # The archived cache fingerprints below verify both final tensors independently.
    device = model.model.embed_tokens.weight.device
    hidden = model.prelude(torch.tensor(prompt, device=device, dtype=torch.long))
    cos, sin = model.model.rotary_emb(hidden, torch.arange(len(prompt), device=device))
    layer = model.model.layers[0]
    norm = layer.input_layernorm(hidden)
    attention = layer.self_attn
    shape = (len(prompt), -1, model.config.head_dim)
    q = attention.q_proj(norm).view(shape)
    raw_k = attention.k_proj(norm)
    raw_v = attention.v_proj(norm)
    _, k = apply_rotary_pos_emb(q, raw_k.view(shape), cos, sin)
    stages = dict(
        embedding=hidden,
        norm=norm,
        k_proj=raw_k,
        v_proj=raw_v,
        rope_cos=cos,
        rope_sin=sin,
        k_write=k,
        v_write=raw_v.view(shape),
    )
    return {
        name: value[:prefix].detach().cpu().contiguous()
        for name, value in stages.items()
        if names is None or name in names
    }


def _compare(left, right):
    if left.shape != right.shape or left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ValueError("expected matching BF16 tensors")
    different = left.view(torch.int16) != right.view(torch.int16)
    a, b = left.double(), right.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    nonfinite_pairs = int((~finite).sum())
    delta = (a - b).abs().masked_fill_(~finite, -1)
    worst = int(delta.argmax())
    worst_abs = float(delta.flatten()[worst])
    return dict(
        elements=left.numel(),
        different_elements=int(different.sum()),
        nonfinite_pairs=nonfinite_pairs,
        max_abs=worst_abs if worst_abs >= 0 else None,
        worst_index=worst if worst_abs > 0 else None,
    )


def _cache_matches(stages, run):
    archived = run["fingerprint"]["sha256"][0][0]
    return {
        name: hashlib.sha256(stages[name + "_write"].view(torch.uint8).numpy()).hexdigest()
        == archived[name]
        for name in ("k", "v")
    }


def layer0_experiment(model, case, boundary):
    p = case.reusable_tokens
    contract = boundary["contract"]
    donor, _, _ = _prompts(
        case, "reuse", contract["output_tokens"], model.config.vocab_size, contract["seed"]
    )
    full = _prefix_stages(model, donor, p)
    chunk = _prefix_stages(model, donor[:p], p)
    matches = {
        "full": _cache_matches(full, boundary["modes"]["reuse"]),
        "chunk": _cache_matches(chunk, boundary["modes"]["aligned"]),
    }
    stats = {name: _compare(value, chunk[name]) for name, value in full.items()}
    # For the largest finite discrepancy in each projection, evaluate that
    # one scalar in FP64 using the same BF16 operands as both device matmuls.
    if stats["norm"]["different_elements"] == 0:
        for name in ("k_proj", "v_proj"):
            index = stats[name]["worst_index"]
            if index is None or stats[name]["nonfinite_pairs"]:
                continue
            token, channel = divmod(index, full[name].shape[1])
            weight = getattr(model.model.layers[0].self_attn, name).weight[channel]
            reference = float(
                torch.dot(full["norm"][token].double(), weight.detach().cpu().double())
            )
            stats[name]["sample"] = dict(
                token=token,
                channel=channel,
                full=float(full[name][token, channel]),
                chunk=float(chunk[name][token, channel]),
                fp64_dot=reference,
            )
    return dict(
        case=case.name,
        full_rows=len(donor),
        chunk_rows=p,
        matched_cache_kv=matches,
        matched_boundary=all(all(pair.values()) for pair in matches.values()),
        stages=stats,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    boundary = json.loads(args.boundary.read_text())
    summary = boundary["summary"]
    if not summary["comparable"] or summary["case"] not in ("total512_p384", "p1024_s8"):
        parser.error("expected a comparable C1 donor-boundary report")
    contract = boundary["contract"]
    if not args.model.is_dir() or checkpoint_identity(args.model) != contract["checkpoint"]:
        parser.error("supply the local checkpoint used by the donor-boundary run")
    if contract["device"] not in ("npu", "cpu") or contract["dtype"] != "bfloat16":
        parser.error("expected a BF16 NPU or CPU boundary report")
    device = resolve_device("npu:0" if contract["device"] == "npu" else "cpu")
    if device.type == "npu":
        torch.npu.set_device(0)
    model = OuroForCausalLM.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    case = next(c for c in suites()["full"] if c.name == summary["case"])
    result = layer0_experiment(model, case, boundary)
    print("LAYER0", json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return int(not result["matched_boundary"])


if __name__ == "__main__":
    raise SystemExit(main())
