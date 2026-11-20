"""Build and exercise the installed CANN MatMulV3 on the original BF16 fixture."""

import argparse
import csv
import json
import os
import shlex
import subprocess
from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from docs.recipes.ascend_bf16_matmul import row_runs

ROWS = (384, 512, 576)


def build(output):
    import sysconfig

    from torch_npu.utils.cpp_extension import NpuExtension

    # Use the include and link settings from the active torch_npu wheel.
    source = Path(__file__).with_suffix(".cpp").resolve()
    ext = NpuExtension("rlt_v3_bench", [str(source)])
    cxx = shlex.split(os.environ.get("CXX") or "c++")
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    subprocess.run(
        [
            *cxx,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            *[f"-I{p}" for p in [*ext.include_dirs, sysconfig.get_path("include")]],
            str(source),
            *[f"-L{p}" for p in ext.library_dirs],
            *[f"-l{lib}" for lib in ext.libraries],
            "-o",
            str(output),
        ],
        check=True,
    )


def load_projection(library):
    import torch_npu  # noqa: F401

    torch.ops.load_library(str(library.resolve(strict=True)))
    return torch.ops.rlt_v3_bench.projection.default


@contextmanager
def use_v3_kv(model, library):
    """With no bridge, observe native Linear calls without replacing their implementation."""
    project = load_projection(library) if library else None
    calls = Counter()
    shapes = set()

    def observe(name, module, inputs):
        (x,) = inputs
        calls[name] += 1
        shapes.add((x.size(0), x.size(1), module.weight.size(0)))

    def forward(module, name, x):
        result = project(x, module.weight)
        observe(name, module, (x,))
        return result

    with ExitStack() as patches:
        for layer in model.model.layers:
            for name in ("k_proj", "v_proj"):
                module = getattr(layer.self_attn, name)
                if module.bias is not None:
                    raise ValueError("V3 experiment requires bias-free K/V projections")
                if project is not None:
                    patches.enter_context(
                        patch.object(module, "forward", partial(forward, module, name))
                    )
                else:
                    patches.callback(
                        module.register_forward_pre_hook(partial(observe, name)).remove
                    )
        yield calls, shapes


def collect(x, weight, project):
    # Check W transpose and output data independently of near-tie rounding.
    control = torch.zeros_like(x[:384])
    control[:, 7] = 1
    actual = project(control, weight).cpu()
    expected = weight[:, 7].cpu().expand_as(actual)
    if not torch.equal(actual, expected):
        raise RuntimeError("one-hot control failed: X @ W.T must select weight[:,7]")

    x = x[2].clone().expand_as(x).contiguous()
    reference = torch.mv(weight.cpu().double(), x[0].cpu().double())
    by_shape = {}
    longest = None
    for m in reversed(ROWS):
        y = project(x[:m], weight).cpu()
        if y.shape != (m, weight.shape[0]) or y.dtype != torch.bfloat16 or not y.isfinite().all():
            raise ValueError(f"invalid projection output at M={m}")
        if longest is None:
            longest = y
        bits = y.view(torch.int16)
        col = y[:, 18]
        by_shape[m] = dict(
            different_elements_vs_first_row=int((bits != bits[:1]).sum()),
            different_elements_vs_576=int((bits != longest[:m].view(torch.int16)).sum()),
            ch18_runs=row_runs(col.tolist()),
            ch18_different_from_rounded_fp64=int((col != reference[18].bfloat16()).sum()),
        )
    return dict(
        fp64_ch18=float(reference[18]),
        rounded_bf16_ch18=float(reference[18].bfloat16()),
        invariant=all(
            not (v["different_elements_vs_first_row"] or v["different_elements_vs_576"])
            for v in by_shape.values()
        ),
        rows=by_shape,
    )


def probe(fixture, library):
    import torch_npu
    from torch_npu.utils.collect_env import get_cann_version

    torch.npu.set_device(0)
    try:
        cann = get_cann_version() if os.environ.get("ASCEND_HOME_PATH") else None
    except (OSError, RuntimeError):
        cann = None
    environment = dict(
        device=torch.npu.get_device_name(0),
        torch=str(torch.__version__),
        torch_npu=torch_npu.__version__,
        cann=cann,
        opp_path=os.environ.get("ASCEND_OPP_PATH"),
        impl="v3" if library else "native",
        ignore_infer_error=os.environ.get("IGNORE_INFER_ERROR"),
    )
    print("PROBE_ENV", json.dumps(environment), flush=True)
    data = torch.load(fixture, map_location="cpu", weights_only=True)
    x = data["cases"]["c4_h4"]["x"]
    w = data["weights"]["v_proj"]
    if (
        x.shape != (576, 2048)
        or w.shape != (2048, 2048)
        or x.dtype != w.dtype
        or w.dtype != torch.bfloat16
    ):
        raise ValueError("expected the archived c4_h4 BF16 linear-inputs fixture")
    op = load_projection(library) if library else torch.nn.functional.linear
    with torch.inference_mode():
        result = collect(x.to("npu:0"), w.to("npu:0"), op)
    return dict(**environment, **result)


def matrix_shapes(path):
    report = json.loads(path.read_text())
    if report["state"] != "complete" or report["contract"].get("kv_projection") not in (
        "MatMulV3",
        "native-dispatch-audit",
    ):
        raise ValueError("expected a completed V3 matrix")
    return sorted(
        {tuple(shape) for entry in report["cases"].values() for shape in entry["v3_shapes"]}
    )


def replay_shapes(library, shapes):
    import torch_npu  # noqa: F401

    project = load_projection(library) if library else torch.nn.functional.linear
    torch.npu.set_device(0)
    with torch.inference_mode():
        for m, k, n in shapes:
            x = torch.zeros((m, k), dtype=torch.bfloat16, device="npu:0")
            weight = x.new_zeros((n, k))
            project(x, weight)
        torch.npu.synchronize()


def evidence(capture, require_v3, shapes=None):
    paths = sorted(capture.rglob("op_summary*.csv"))
    shapes = shapes or [(m, 2048, 2048) for m in ROWS]
    seen = {shape: set() for shape in shapes}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                kind = row.get("op type", "")
                if "matmul" not in kind.lower() and "gemm" not in kind.lower():
                    continue
                inputs = row.get("input shapes", "").strip('"').replace(" ", "").split(";")
                output = row.get("output shapes", "").strip('" ()[]').replace(" ", "")
                dtypes = row.get("input data types", "").strip('"').upper().split(";")
                for m, k, n in shapes:
                    if (
                        inputs[:1] == [f"{m},{k}"]
                        and len(inputs) >= 2
                        and inputs[1] in (f"{n},{k}", f"{k},{n}")
                        and output == f"{m},{n}"
                    ):
                        bf16 = {"DT_BF16", "BF16", "BFLOAT16", "DT_BFLOAT16"}
                        if (
                            len(dtypes) < 2
                            or any(t.strip() not in bf16 for t in dtypes[:2])
                            or row.get("output data types", "").strip('"').upper() not in bf16
                        ):
                            raise ValueError(
                                f"expected BF16 operands and output for {kind} at M={m}"
                            )
                        seen[m, k, n].add(kind)
    return dict(
        verified=all(
            kinds and (not require_v3 or kinds == {"MatMulV3"}) for kinds in seen.values()
        ),
        summaries=[str(p) for p in paths],
        types={f"{m}x{k}x{n}": sorted(v) for (m, k, n), v in seen.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("build").add_argument("library", type=Path)
    run = sub.add_parser("probe")
    run.add_argument("fixture", type=Path)
    run.add_argument("--library", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("capture", type=Path)
    verify.add_argument("--v3", action="store_true")
    verify.add_argument("--matrix", type=Path)
    shapes = sub.add_parser("replay-shapes")
    shapes.add_argument("library", type=Path)
    shapes.add_argument("matrix", type=Path)
    for p in (run, verify):
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "build":
        build(args.library)
        return
    if args.mode == "replay-shapes":
        replay_shapes(
            None if str(args.library) == "-" else args.library, matrix_shapes(args.matrix)
        )
        return
    if args.mode == "probe":
        result = probe(args.fixture, args.library)
    else:
        result = evidence(
            args.capture, args.v3, matrix_shapes(args.matrix) if args.matrix else None
        )
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(args.mode.upper(), json.dumps(result, allow_nan=False), flush=True)
    if args.mode == "verify" and not result["verified"]:
        parser.exit(1, f"Device dispatch unverified. Observed types are saved in {args.out}\n")


if __name__ == "__main__":
    main()
