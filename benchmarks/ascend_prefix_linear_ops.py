"""Replay BF16 linear projections on fixed operands in PyTorch on CPU or NPU."""

import hashlib
import math
import time
from statistics import median

import torch
import torch.nn.functional as F


def _metadata(tensor):
    data = dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        storage_offset=tensor.storage_offset(),
        data_ptr=hex(tensor.data_ptr()),
        contiguous=tensor.is_contiguous(),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )
    if tensor.device.type == "npu":
        import torch_npu

        get_format = getattr(torch_npu, "get_npu_format", None)
        if callable(get_format):
            try:
                data["npu_format"] = str(get_format(tensor))
            except RuntimeError as exc:
                data["npu_format_error"] = str(exc)
    return data


def _bits_different(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def _sha256(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def _finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _compare_values(a, b, inputs, weight):
    # All four tensors are on CPU. Keep bit counts distinct from numeric deltas
    # (e.g. signed zeros), and use the real BF16 operands for the FP64 reference.
    different = a.view(torch.int16) != b.view(torch.int16)
    count = int(different.sum())
    aa, bb = a.float(), b.float()
    finite = torch.isfinite(aa) & torch.isfinite(bb)
    delta = (aa.double() - bb.double()).abs().masked_fill_(~finite, -1)
    max_abs = float(delta.max())
    sample = None
    if count:
        index = int(different.flatten().to(torch.int32).argmax())
        row, channel = divmod(index, a.shape[1])
        sample = dict(
            row=row,
            channel=channel,
            reference=_finite_scalar(a[row, channel]),
            actual=_finite_scalar(b[row, channel]),
            fp64_dot=_finite_scalar(torch.dot(inputs[row].double(), weight[channel].double())),
        )
    return dict(
        elements=a.numel(),
        different_elements=count,
        nonfinite_pairs=int((~finite).sum()),
        max_abs=max_abs if max_abs >= 0 else None,
        sample=sample,
    )


def _projection(op, weight, tile_rows):
    if op == "linear":
        return lambda x: F.linear(x, weight)
    if op == "mm":
        return lambda x: torch.mm(x, weight.T)
    if op == "npu_linear":
        if weight.device.type != "npu":
            raise ValueError("npu_linear requires NPU tensors")
        import torch_npu

        if not callable(getattr(torch_npu, "npu_linear", None)):
            raise ValueError("this torch_npu build does not provide npu_linear")
        return lambda x: torch_npu.npu_linear(x, weight)
    if op != "tile64":
        raise ValueError(f"unknown projection: {op}")

    def tiled(x):
        outputs = []
        for tile in x.split(tile_rows):
            active = tile.shape[0]
            if active < tile_rows:
                tile = torch.cat((tile, tile.new_zeros((tile_rows - active, tile.shape[1]))))
            outputs.append(F.linear(tile.contiguous(), weight)[:active])
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    return tiled


@torch.inference_mode()
def measure_linear(
    x, weight, *, case, prefix_rows, slot_rows=None, repeats=3, tile_rows=64, op="linear", warmups=1
):
    """Compare full/short M (C1) or four copies of one slot at a fixed M (C4).

    C4's primary counts cover the shared prefix, with slot 2 as reference.
    Work and timing refer to a single complete projection of each input shape.
    """
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or not x.is_contiguous()
        or min(*x.shape, *weight.shape) < 1
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.device != weight.device
        or x.device.type not in ("cpu", "npu")
    ):
        raise ValueError("expected contiguous 2D BF16 input and matching BF16 weight on CPU or NPU")
    if repeats < 2 or warmups < 0 or tile_rows < 1:
        raise ValueError(
            "supply at least two measured repeats, nonnegative warmups and positive tile rows"
        )
    if slot_rows is None:
        if not 0 < prefix_rows < x.shape[0]:
            raise ValueError("C1 requires a nonempty strict prefix of the full matrix")
        geometries = {"full": x, "prefix": x[:prefix_rows]}
    else:
        if not 0 < prefix_rows <= slot_rows or x.shape[0] != 4 * slot_rows:
            raise ValueError("C4 requires four equal-sized slots and a valid shared prefix length")
        geometries = {"full": x}

    project = _projection(op, weight, tile_rows)

    # Layout and initial tensors are held fixed, including the C1 prefix's backing storage.
    # Only the call and completion barrier are timed. CPU transfers/bit checks are outside.
    def synchronize():
        if x.device.type == "npu":
            torch.npu.synchronize(x.device)

    for _ in range(warmups):
        for values in geometries.values():
            project(values)
    synchronize()

    controls, first, timings, repeat_counts = [], {}, {}, {}
    for it in range(repeats):
        # Alternate which C1 shape is evaluated first.
        order = list(geometries) if it % 2 == 0 else list(reversed(geometries))
        for name in order:
            values = geometries[name]
            synchronize()
            start = time.perf_counter()
            output = project(values)
            synchronize()
            ms = (time.perf_counter() - start) * 1000
            if output.shape != (values.shape[0], weight.shape[0]) or output.dtype != x.dtype:
                raise ValueError(f"unexpected {op} output dtype or geometry")
            host = output.detach().to(device="cpu", copy=True).contiguous()
            timings.setdefault(name, []).append(ms)
            if it == 0:
                first[name] = host
                repeat_counts[name] = []
                if not torch.isfinite(host).all():
                    controls.append(f"{name}:nonfinite_output")
            else:
                repeat_counts[name].append(_bits_different(first[name], host))
    controls.extend(f"{name}:repeat_drift" for name, counts in repeat_counts.items() if any(counts))
    host_x, host_w = x.detach().cpu(), weight.detach().cpu()
    whole_slot_counts = input_slot_counts = None
    if slot_rows is None:
        prefixes = [first["full"][:prefix_rows], first["prefix"]]
        comparisons = [_compare_values(prefixes[0], prefixes[1], host_x, host_w)]
    else:
        slots = first["full"].reshape(4, slot_rows, -1)
        inputs = host_x.reshape(4, slot_rows, -1)
        focus = 2
        input_slot_counts = [_bits_different(inputs[focus], slot) for slot in inputs]
        if any(input_slot_counts):
            controls.append("slot_inputs_not_identical")
        prefixes = [slot[:prefix_rows] for slot in slots]
        comparisons = [
            _compare_values(prefixes[focus], block, inputs[focus], host_w) for block in prefixes
        ]
        whole_slot_counts = [_bits_different(slots[focus], slot) for slot in slots]

    work = {}
    for name, values in geometries.items():
        rows = values.shape[0]
        calls = (rows + tile_rows - 1) // tile_rows if op == "tile64" else 1
        work[name] = dict(
            calls=calls,
            active_rows=rows,
            padded_rows=calls * tile_rows - rows if op == "tile64" else 0,
        )
    return dict(
        case=case,
        op=op,
        comparable=not controls,
        controls=controls,
        geometry={name: _metadata(v) for name, v in geometries.items()},
        weight=_metadata(weight),
        repeats=repeats,
        warmups=warmups,
        tile_rows=tile_rows if op == "tile64" else None,
        prefix_rows=prefix_rows,
        slot_rows=slot_rows,
        work=work,
        repeat_different_elements=repeat_counts,
        timing_ms={name: dict(samples=ts, p50=median(ts)) for name, ts in timings.items()},
        first_output_sha256={name: _sha256(v) for name, v in first.items()},
        prefix_sha256=[_sha256(block) for block in prefixes],
        different_elements=[c["different_elements"] for c in comparisons],
        comparison=comparisons,
        input_slot_different_elements=input_slot_counts,
        whole_slot_different_elements=whole_slot_counts,
    )
