"""Replay an F.linear boundary. Requires only PyTorch (and torch_npu on Ascend)."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def same(a, b):
    if a is None or b is None:
        return a is b
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),
            b.detach().cpu().contiguous().view(torch.uint8),
        )
    )


def pack(t):
    if t is None:
        return None
    fmt = None
    if t.device.type == "npu":
        import torch_npu

        fmt = torch_npu.get_npu_format(t)
    return dict(
        values=t.detach().cpu().contiguous().clone(),
        shape=list(t.shape),
        stride=list(t.stride()),
        offset=t.storage_offset(),
        dtype=str(t.dtype),
        npu_format=fmt,
    )


def restore(blob, device):
    if blob is None:
        return None
    t = blob["values"]
    n = (
        1
        + blob["offset"]
        + sum((d - 1) * s for d, s in zip(blob["shape"], blob["stride"], strict=True))
    )
    base = torch.empty(n, dtype=t.dtype, device=device)
    x = base.as_strided(blob["shape"], blob["stride"], blob["offset"])
    x.copy_(t)
    if device.startswith("npu") and blob["npu_format"] is not None:
        import torch_npu

        if torch_npu.get_npu_format(x) != blob["npu_format"]:
            x = torch_npu.npu_format_cast(x, blob["npu_format"])
        if torch_npu.get_npu_format(x) != blob["npu_format"]:
            raise RuntimeError("could not restore physical NPU format")
    if list(x.stride()) != blob["stride"] or x.storage_offset() != blob["offset"]:
        raise RuntimeError("could not restore original tensor layout")
    return x


def row(t, i):
    return t.reshape(-1, t.shape[-1])[i]


def one(blob, device):
    a, b, bias = (restore(blob[k], device) for k in ("input", "weight", "bias"))
    y = F.linear(a, b, bias).detach().cpu()
    i = blob["row"]
    src, w, bias_cpu = blob["input"]["values"], blob["weight"]["values"], blob["bias"]
    ref = F.linear(
        row(src, i).double(), w.double(), bias_cpu["values"].double() if bias_cpu else None
    )
    actual = row(y, i)
    recorded = row(blob["output"]["values"], i)
    rounded = ref.to(actual.dtype)
    error = (recorded.double() - rounded.double()).abs()
    channel = int(error.argmax())
    return (
        actual,
        recorded,
        dict(
            full_output_bitwise=same(y, blob["output"]["values"]),
            row_bitwise=same(actual, recorded),
            max_abs_vs_fp64=(recorded.double() - ref).abs().max().item(),
            max_abs_vs_rounded=error.max().item(),
            channels_different_from_rounded=int((recorded != rounded).sum()),
            worst_channel=channel,
            recorded=float(recorded[channel]),
            fp64=float(ref[channel]),
            rounded=float(rounded[channel]),
        ),
    )


def replay(left, right, device):
    for k in ("call", "request", "pos"):
        if left[k] != right[k]:
            raise ValueError(f"unmatched logical coordinates: {k}")
    rows = [one(blob, device) for blob in (left, right)]
    inputs_equal = same(
        row(left["input"]["values"], left["row"]), row(right["input"]["values"], right["row"])
    )
    weights_equal = same(left["weight"]["values"], right["weight"]["values"])
    bias_equal = same((left["bias"] or {}).get("values"), (right["bias"] or {}).get("values"))
    diff = not same(rows[0][1], rows[1][1])
    reproduced = diff and all(x[2]["full_output_bitwise"] for x in rows)
    return dict(
        call=left["call"],
        request=left["request"],
        position=left["pos"],
        device=device,
        input_row_bitwise=inputs_equal,
        weight_bitwise=weights_equal,
        bias_bitwise=bias_equal,
        captured_row_differs=diff,
        independent_reproduction="REPRODUCED" if reproduced else "NOT_REPRODUCED",
        same_input_divergence=reproduced and inputs_equal and weights_equal and bias_equal,
        off=rows[0][2],
        reuse=rows[1][2],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("off", type=Path)
    p.add_argument("reuse", type=Path)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--only", choices=["off", "reuse"], help="one full-sized NPU dispatch for msprof op"
    )
    args = p.parse_args()
    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
    if args.only:
        blob = torch.load(
            args.off if args.only == "off" else args.reuse, map_location="cpu", weights_only=True
        )
        x, w, b = (restore(blob[k], args.device) for k in ("input", "weight", "bias"))
        F.linear(x, w, b)
        if args.device.startswith("npu"):
            torch.npu.synchronize()
    else:
        result = replay(
            *(
                torch.load(path, map_location="cpu", weights_only=True)
                for path in (args.off, args.reuse)
            ),
            args.device,
        )
        value = json.dumps(result, indent=2, allow_nan=False)
        if args.out:
            args.out.write_text(value + "\n")
        print(value)
