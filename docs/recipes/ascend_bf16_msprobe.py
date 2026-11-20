"""Dump the real Ouro V projection with msprobe (MindStudio 8.2.RC1)."""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def linear_record(directory):
    indexes = list(directory.rglob("dump.json"))
    if len(indexes) != 1:
        raise RuntimeError(f"Expected one L1 dump.json in {directory}, found {len(indexes)}")
    dump = json.loads(indexes[0].read_text())
    names = [key for key in dump["data"] if re.fullmatch(r"Functional\.linear\.\d+\.forward", key)]
    if len(names) != 1:
        raise RuntimeError(f"Expected one forward F.linear in {indexes[0]}, found {names}")
    return names[0], dump["data"][names[0]], indexes[0].parent / "dump_tensor_data"


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def same_bf16(a, b):
    return a.dtype == b.dtype == torch.bfloat16 and torch.equal(
        a.view(torch.int16), b.view(torch.int16)
    )


def summarize(directory, row, weight, result):
    name, record, tensors = linear_record(directory)
    x, w = [load_tensor(tensors / arg["data_name"]) for arg in record["input_args"][:2]]
    y = load_tensor(tensors / record["output"][0]["data_name"])
    rows = result.shape[0]
    if not same_bf16(x, row.expand(rows, -1)) or not same_bf16(w, weight):
        raise RuntimeError("Dumped inputs differ from the fixture row or V weight")
    if not same_bf16(y, result):
        raise RuntimeError("Dumped output differs from the returned BF16 projection")
    values = y[:, 18]
    reference = torch.dot(x[2].double(), w[18].double())
    return dict(
        api=name,
        rows=rows,
        inputs_match_fixture=True,
        output_matches_call=True,
        channel=18,
        row2=values[2].item(),
        row290=values[290].item(),
        rows_different_from_row2=(values != values[2]).sum().item(),
        fp64_dot=reference.item(),
        rounded_bf16=reference.bfloat16().item(),
    )


@torch.inference_mode()
def collect(fixture, rows, out, level, device):
    from msprobe.pytorch import PrecisionDebugger

    data = load_tensor(fixture)
    host_x = data["cases"]["c4_h4"]["x"]
    host_w = data["weights"]["v_proj"]
    if host_x.shape != (576, 2048) or host_w.shape != (2048, 2048):
        raise ValueError("Expected the 576-row C4 fixture and the complete 2048 x 2048 V weight")
    if host_x.dtype != torch.bfloat16 or host_w.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 fixture tensors")
    case = out.resolve() / f"m{rows}"
    # The L2 list must name the exact invocation observed in the matching L1 dump.
    api = "linear" if level == "L1" else linear_record(case / "L1")[0]
    directory = case / level
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(task="tensor", level=level, dump_path=str(directory), tensor={"list": [api]})
    if level == "L1":
        config["tensor"]["data_mode"] = ["forward"]
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    weight = host_w.to(device)
    x = host_x.to(device, copy=True)
    x.copy_(x[2].clone().expand_as(x))
    inputs = x[:rows]
    debugger = PrecisionDebugger(config_path=str(config_path))
    debugger.start()
    try:
        projected = F.linear(inputs, weight)
    finally:
        debugger.stop()
    debugger.step()

    if level == "L1":
        report = summarize(directory, host_x[2], host_w, projected.cpu())
    else:
        # The collector may log success even if the runtime emitted no kernel tensor data.
        files = sorted(
            p
            for p in (directory / "step0").rglob("*")
            if p.is_file()
            and "matmul" in p.name.lower()
            and p.suffix not in {".json", ".log"}
            and p.stat().st_size > 0
        )
        if not files:
            raise RuntimeError(
                f"No MatMul kernel dump in {directory / 'step0'}; check the msprobe log"
            )
        report = dict(
            api=api, rows=rows, kernel_files=[str(p.relative_to(directory)) for p in files]
        )
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MSPROBE_{level}", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(384, 512), default=512)
    parser.add_argument("--level", choices=("L1", "L2"), default="L1")
    args = parser.parse_args()
    import torch_npu  # noqa: F401 -- registers the NPU backend

    torch.npu.set_device(0)
    collect(args.fixture, args.rows, args.out, args.level, "npu:0")


if __name__ == "__main__":
    main()
