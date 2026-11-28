#!/usr/bin/env python3
"""Standalone Ouro-shaped paged prefill probe. Use --help, self-test, or run."""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HEADS, DIM = 16, 128
LAYERS, LAYER = 24, 7
ATOL, RTOL = 0.016, 0.02
CASES = {"w2": ([32] * 32, 128, 96), "c4_h1": ([16, 144, 144, 144], 144, 0)}
PAGES = (128, 16)
LAYOUTS = ("compact", "rlt")
SCRIPT = Path(__file__).resolve()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def logical_inputs(case, seed):
    lengths, kv_len, shared = CASES[case]
    gen = torch.Generator().manual_seed(seed)

    def bf16(shape):
        return torch.randn(shape, generator=gen).to(torch.bfloat16)

    q = bf16((sum(lengths), HEADS, DIM))
    k = bf16((len(lengths), kv_len, HEADS, DIM))
    v = bf16(k.shape)
    # Mark logical positions and requests in two channels, in addition to random data.
    v[:, :, :, 0] = torch.arange(kv_len)[None, :, None] / 32
    v[:, :, :, 1] = torch.arange(len(lengths))[:, None, None] / 8
    if shared:
        k[1:, :shared] = k[0, :shared].clone()
        v[1:, :shared] = v[0, :shared].clone()
    return q, k, v


def digest(*tensors):
    sha = hashlib.sha256()
    for tensor in tensors:
        sha.update(tensor.contiguous().view(torch.uint8).numpy().data)
    return sha.hexdigest()


def reference(q, k, v, lengths):
    """Compute each request separately, directly from logical BF16 inputs in FP32."""
    outputs = []
    for row, query in enumerate(q.split(lengths)):
        qlen, klen = len(query), k.shape[1]
        visible = torch.arange(klen)[None, :] <= (klen - qlen + torch.arange(qlen))[:, None]
        logits = query.float().transpose(0, 1) @ k[row].float().transpose(0, 1).transpose(1, 2)
        probs = (logits / math.sqrt(DIM)).masked_fill(~visible, -torch.inf).softmax(dim=-1)
        outputs.append((probs @ v[row].float().transpose(0, 1)).transpose(0, 1))
    return torch.cat(outputs)


def pack_pages(k, v, page, shared, seed):
    batch, klen = k.shape[:2]
    width = math.ceil(klen / page)
    # Reuse only full common-prefix pages. At page=128 W2 keeps the same logical
    # prefix, inside a private physical page for each request.
    owners = [
        [(0 if (col + 1) * page <= shared else row, col) for col in range(width)]
        for row in range(batch)
    ]
    unique = list(dict.fromkeys(itertools.chain.from_iterable(owners)))
    pool_size = max(batch * width + 2, 2 * len(unique) + 3)
    available = list(range(1, pool_size, 2))
    random.Random(seed + page).shuffle(available)
    ids = dict(zip(unique, available[: len(unique)], strict=True))
    table = torch.tensor([[ids[o] for o in seq] for seq in owners], dtype=torch.int32)
    # Spare and unused tail slots deliberately contain finite poison.
    shape = (pool_size, page, HEADS * DIM)
    pk, pv = torch.full(shape, 5.0, dtype=k.dtype), torch.full(shape, -7.0, dtype=v.dtype)
    for (row, col), physical in ids.items():
        start = col * page
        end = min(start + page, klen)
        pk[physical, : end - start].copy_(k[row, start:end].flatten(1))
        pv[physical, : end - start].copy_(v[row, start:end].flatten(1))
    return pk, pv, table


def make_pools(pk, pv, layout, device, layers=LAYERS, selected=LAYER):
    depth = layers if layout == "rlt" else 1  # number of model layers stored per block
    index = selected if layout == "rlt" else 0
    pools = []
    for packed, poison in ((pk, 3.0), (pv, -4.0)):
        blocks, page, _ = packed.shape
        pool = torch.empty((blocks, depth, page, HEADS, DIM), dtype=packed.dtype, device=device)
        if depth > 1:
            pool.fill_(poison)
            pool[:, index - 1].fill_(-poison)
        # One-time setup; the measured/profiled operator window starts afterwards.
        pool[:, index].copy_(packed.view(blocks, page, HEADS, DIM).to(device))
        pools.append(pool)
    return pools[0], pools[1], index


def operator_inputs(kpool, vpool, table, layer):
    blocks, layers, page, _, _ = kpool.shape
    # A row b of the original page table addresses (b, layer) in the base pool.
    # Folding only adjoining storage dimensions keeps both K and V contiguous.
    k = kpool.view(blocks * layers, page, HEADS * DIM)
    v = vpool.view_as(k)
    mapped = table if layers == 1 else table.mul(layers).add_(layer)
    if k.data_ptr() != kpool.data_ptr() or v.data_ptr() != vpool.data_ptr():
        raise RuntimeError("K/V views do not alias their base pools")
    return k, v, mapped


def cpu_readback(q, k, v, table, lengths, kv_len):
    """Independent paged readback for the CPU self-test. Use SDPA with explicit mask."""
    outputs = []
    for row, query in enumerate(q.split(lengths)):

        def read(pool):
            return pool.index_select(0, table[row].long()).reshape(-1, HEADS, DIM)[:kv_len]

        start = kv_len - len(query)
        mask = torch.ones(len(query), kv_len, dtype=torch.bool).tril(diagonal=start)
        output = torch.nn.functional.scaled_dot_product_attention(
            query.float().transpose(0, 1),
            read(k).float().transpose(0, 1),
            read(v).float().transpose(0, 1),
            attn_mask=mask,
        )
        outputs.append(output.transpose(0, 1))
    return torch.cat(outputs)


def compare(actual, expected, lengths):
    if actual.shape != expected.shape:
        raise ValueError(f"output shape {actual.shape} differs from {expected.shape}")
    reports = []
    for row, (out, ref) in enumerate(
        zip(actual.float().split(lengths), expected.split(lengths), strict=True)
    ):
        err = (out - ref).abs()
        bad = ~torch.isfinite(out) | (err > ATOL + RTOL * ref.abs())
        finite = torch.isfinite(err).all().item()
        first = bad.nonzero()[0].tolist() if bad.any() else None
        reports.append(
            dict(
                request=row,
                q_len=len(out),
                bad_elements=int(bad.sum()),
                first_bad_q_head_channel=first,
                max_abs=float(err.max()) if finite else None,
                first_q_max_abs=float(err[0].max()) if finite else None,
                rms=float(err.square().mean().sqrt()) if finite else None,
            )
        )
    return reports


def inspect_trace(path, full_pool_bytes, layer_bytes):
    payload = json.loads(path.read_text())
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    device_events = 0
    copies = []
    device_pids = {
        e.get("pid")
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and "ascend hardware" in str(e.get("args", {}).get("name", "")).lower()
    }
    for e in events:
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", ""))
        if e.get("ph") == "X" and (
            cat in {"kernel", "memcpy", "memset"} or e.get("pid") in device_pids
        ):
            device_events += 1
        if e.get("ph") != "X" or not any(
            term in name.lower()
            for term in ("memcpy", "contiguous", "clone", "transdata", "copy_", "tensorcopy")
        ):
            continue
        args = e.get("args", {})
        sizes = [
            int(val)
            for key, val in args.items()
            if key.lower().strip() in {"bytes", "size", "size(bytes)", "size (bytes)", "num bytes"}
            and str(val).isdigit()
        ]
        copies.append(
            dict(
                name=name,
                category=cat,
                bytes=max(sizes, default=None),
                shapes=args.get("Input Dims", args.get("Input Shapes")),
            )
        )
    large = [c for c in copies if c["bytes"] is not None and c["bytes"] >= full_pool_bytes]
    return dict(
        trace=str(path.name),
        device_events=device_events,
        copy_events=copies,
        transfers_at_least_one_full_pool=large,
        transfers_at_least_one_layer=[c for c in copies if (c["bytes"] or 0) >= layer_bytes],
    )


def profile_call(torch_npu, call, path, expected_out, pool_bytes, layer_bytes):
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        output = call()
        torch.npu.synchronize()
    prof.export_chrome_trace(str(path))
    audit = inspect_trace(path, pool_bytes, layer_bytes)
    audit["matches_unprofiled_output"] = torch.equal(output.cpu(), expected_out)
    return audit


def environment():
    versions = {}
    for root in (
        os.getenv("ASCEND_HOME_PATH"),
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ):
        if root:
            for path in (
                Path(root) / "version.cfg",
                Path(root) / "version.info",
                Path(root) / platform.machine().replace("amd64", "x86_64") / "version.info",
                Path(root) / (platform.machine() + "-linux") / "ascend_toolkit_install.info",
            ):
                if path.is_file():
                    versions[str(path)] = path.read_text(errors="replace")[:2000]
    return dict(
        torch=str(torch.__version__),
        python=platform.python_version(),
        cann_files=versions,
        visibility=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        local_device=0,
    )


def trial(a, record):
    import torch_npu

    try:
        from torch_npu.utils.collect_env import get_cann_version

        record["environment"]["cann_version"] = str(get_cann_version())
    except (ImportError, OSError, RuntimeError) as exc:
        record["environment"]["cann_version_query"] = str(exc)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    record["environment"].update(
        torch_npu=str(torch_npu.__version__),
        device_name=torch.npu.get_device_name(0),
        allow_hf32=bool(torch.npu.matmul.allow_hf32),
    )
    lengths, kv_len, shared = CASES[a.case]
    q, k, v = logical_inputs(a.case, a.seed)
    ends = list(itertools.accumulate(lengths))
    record.update(
        logical_sha256=digest(q, k, v),
        actual_seq_lengths=ends,
        actual_seq_lengths_kv=[kv_len] * len(lengths),
        scale=DIM**-0.5,
        dtype="BF16",
        heads=HEADS,
        head_dim=DIM,
        sparse_mode=3,
        inner_precise=0,
        tolerance=dict(atol=ATOL, rtol=RTOL),
    )
    expected = reference(q, k, v, lengths)
    pk, pv, table = pack_pages(k, v, a.page, shared, a.seed)
    device = torch.device("npu:0")
    kpool, vpool, layer = make_pools(pk, pv, a.layout, device)
    table_dev, query = table.to(device), q.to(device)
    # sparse_mode=3 uses CANN's compressed 2048x2048 mask. True means blocked.
    mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).to(device)
    kv, vv, mapped = operator_inputs(kpool, vpool, table_dev, layer)
    pool_bytes = kpool.numel() * kpool.element_size()
    record.update(
        page_table=table.tolist(),
        operator_page_table=mapped.cpu().tolist(),
        pool_shape=list(kpool.shape),
        pool_stride=list(kpool.stride()),
        operator_kv_shape=list(kv.shape),
        kv_view_stride=list(kv.stride()),
        selected_layer=layer,
        k_alias=kv.data_ptr() == kpool.data_ptr(),
        v_alias=vv.data_ptr() == vpool.data_ptr(),
        one_full_k_pool_bytes=pool_bytes,
        one_layer_k_bytes=pool_bytes // kpool.shape[1],
    )
    del kv, vv, mapped

    def call():
        key, value, pages = operator_inputs(kpool, vpool, table_dev, layer)
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=HEADS,
            num_key_value_heads=HEADS,
            input_layout="TND",
            scale=DIM**-0.5,
            block_table=pages,
            block_size=a.page,
            sparse_mode=3,
            inner_precise=0,
            atten_mask=mask,
            actual_seq_lengths=ends,
            actual_seq_lengths_kv=[kv_len] * len(lengths),
        )
        return output

    torch.npu.synchronize()
    warmup_output = call()  # warm the same shape before measuring allocator growth
    torch.npu.synchronize()
    del warmup_output
    torch.npu.reset_peak_memory_stats()
    before = torch.npu.memory_allocated()
    result = call()
    torch.npu.synchronize()
    record["extra_peak_allocated_bytes"] = torch.npu.max_memory_allocated() - before
    actual = result.cpu()
    if actual.dtype != torch.bfloat16:
        raise ValueError(f"unexpected FIA output dtype {actual.dtype}")
    record["requests"] = compare(actual, expected, lengths)
    record["output_sha256"] = digest(actual)
    record["status"] = "FAIL" if any(r["bad_elements"] for r in record["requests"]) else "PASS"
    if a.profile:
        try:
            record["profile"] = profile_call(
                torch_npu,
                call,
                Path.cwd() / "trace.json",
                actual,
                pool_bytes,
                record["one_layer_k_bytes"],
            )
        except Exception:
            record["profile"] = {"error": traceback.format_exc()}


def self_test(seed):
    checks = 0
    for case, (lengths, kv_len, shared) in CASES.items():
        q, k, v = logical_inputs(case, seed)
        expected = reference(q, k, v, lengths)
        # With an upper-left causal mask the hot request's first query sees K[0] alone.
        if not compare(v[0, :1].float(), expected[:1], [1])[0]["bad_elements"]:
            raise AssertionError("wrong causal alignment escaped detection")
        for page, layout in itertools.product(PAGES, LAYOUTS):
            pk, pv, ids = pack_pages(k, v, page, shared, seed)
            kp, vp, layer = make_pools(pk, pv, layout, "cpu", layers=3, selected=1)
            key, value, table = operator_inputs(kp, vp, ids, layer)
            out = cpu_readback(q, key, value, table, lengths, kv_len)
            torch.testing.assert_close(out, expected, atol=3e-5, rtol=3e-5)
            if (ids[:, 0].unique().numel() == 1) != (shared >= page):
                raise AssertionError("incorrect physical shared-prefix mapping")
            corrupt = table.clone()
            corrupt[0, -1] = 0  # unused physical page filled with poison
            wrong = cpu_readback(q, key, value, corrupt, lengths, kv_len)
            if not compare(wrong, expected, lengths)[0]["bad_elements"]:
                raise AssertionError("incorrect page mapping escaped detection")
            if layout == "rlt":
                wrong_layer = cpu_readback(q, key, value, table - 1, lengths, kv_len)
                if not all(r["bad_elements"] for r in compare(wrong_layer, expected, lengths)):
                    raise AssertionError("incorrect model-layer mapping escaped detection")
            checks += 1
    print(f"PASS: {checks} CPU fixtures; alias, wrong mask, wrong page and wrong layer checked.")


def run_card(a, folder, card, profile_card):
    records = []
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(card))
    for case, page, layout in itertools.product(CASES, PAGES, LAYOUTS):
        work = folder / f"card-{card}" / f"{case}-{page}-{layout}"
        work.mkdir(parents=True)
        profile = card == profile_card and page == 16 and layout == "rlt"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "_trial",
            "--case",
            case,
            "--page",
            str(page),
            "--layout",
            layout,
            "--seed",
            str(a.seed),
        ]
        if profile:
            cmd.append("--profile")
        row = dict(
            case=case,
            page=page,
            layout=layout,
            card=card,
            status="ERROR",
            file=str((work / "result.json").relative_to(folder)),
            profile_requested=profile,
        )
        try:
            with (work / "run.log").open("w") as log:
                proc = subprocess.run(
                    cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout
                )
            if (work / "result.json").exists():
                row.update(json.loads((work / "result.json").read_text()))
            if proc.returncode != 0 and row["status"] not in {"FAIL", "ERROR"}:
                row.update(status="ERROR", error=f"child exited {proc.returncode}")
            if row["status"] == "ERROR" and "error" not in row:
                row["error"] = f"child exited {proc.returncode}; see run.log"
        except Exception as exc:
            row.update(status="ERROR", error=str(exc))
        if not (work / "result.json").exists():
            write_json(work / "result.json", row)
        records.append(row)
        print(f"card {card} {case} page={page} {layout}: {row['status']}", flush=True)
    return records


def summarize(records, args, folder):
    pairs = []
    for card, case, layout in itertools.product(args.devices, CASES, LAYOUTS):
        rows = {
            r["page"]: r
            for r in records
            if (r["card"], r["case"], r["layout"]) == (card, case, layout)
        }
        control, target = rows[128], rows[16]
        if control["status"] != "PASS":
            state = "CONTROL_FAILED"
        elif target["status"] != "PASS":
            state = "TARGET_FAILED"
        elif control["logical_sha256"] != target["logical_sha256"]:
            state = "INPUT_MISMATCH"
        else:
            state = "PASS"
        pairs.append(
            dict(
                card=card,
                case=case,
                layout=layout,
                status=state,
                page128=control["status"],
                page16=target["status"],
            )
        )
    hashes = [{r.get("logical_sha256") for r in records if r["case"] == c} for c in CASES]
    same_inputs = all(len(values) == 1 and None not in values for values in hashes)
    numerics = (
        "PASS"
        if same_inputs and all(p["status"] == "PASS" for p in pairs)
        else "INCOMPLETE_OR_FAIL"
    )
    audits = []
    for r in records:
        if not r["profile_requested"]:
            continue
        p = r.get("profile", {})
        peak = r.get("extra_peak_allocated_bytes")
        layer_bytes = r.get("one_layer_k_bytes")
        if p.get("error") or not p.get("device_events") or not p.get("matches_unprofiled_output"):
            status = "INCONCLUSIVE"
        elif p["transfers_at_least_one_full_pool"]:
            status = "LARGE_COPY_OBSERVED"
        elif p["transfers_at_least_one_layer"]:
            status = "LAYER_SIZED_COPY_OBSERVED"
        elif (
            r.get("k_alias")
            and r.get("v_alias")
            and peak is not None
            and layer_bytes is not None
            and peak < layer_bytes
            and all(c["bytes"] is not None for c in p["copy_events"])
        ):
            status = "NO_POOL_SIZED_COPY_OBSERVED"
        else:
            status = "REVIEW_TRACE"  # transfer sizes / internal workspaces may be unreported
        audits.append(
            dict(
                case=r["case"],
                card=r["card"],
                status=status,
                file=r["file"],
                peak_delta=r.get("extra_peak_allocated_bytes"),
                pool_bytes=r.get("one_full_k_pool_bytes"),
                layer_bytes=r.get("one_layer_k_bytes"),
            )
        )
    write_json(
        folder / "summary.json",
        dict(
            machine=args.machine,
            seed=args.seed,
            devices=args.devices,
            script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            numeric_status=numerics,
            same_logical_inputs=same_inputs,
            pairs=pairs,
            copy_audits=audits,
            records=[
                {k: r[k] for k in ("card", "case", "page", "layout", "status", "file")}
                for r in records
            ],
        ),
    )
    print(f"{numerics} | {folder / 'summary.json'} | copy audit: {audits or 'not collected'}")
    capture_ok = all(a["status"] in {"REVIEW_TRACE", "NO_POOL_SIZED_COPY_OBSERVED"} for a in audits)
    return 0 if numerics == "PASS" and capture_ok else 1


def devices_arg(text):
    try:
        cards = [int(x) for x in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("use comma-separated numeric device IDs") from e
    if min(cards) < 0 or len(set(cards)) != len(cards):
        raise argparse.ArgumentTypeError("use distinct nonnegative device IDs")
    return cards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="one independent worker per selected physical NPU")
    run.add_argument("--machine", required=True)
    run.add_argument("--out", type=Path, required=True, help="a new results directory")
    run.add_argument("--devices", type=devices_arg, default=list(range(8)))
    run.add_argument("--timeout", type=int, default=600, help="seconds per isolated trial")
    run.add_argument("--no-profile", action="store_true", help="skip the separate profiled replays")
    test = sub.add_parser(
        "self-test", help="CPU-only test of fixtures, reference and address mapping"
    )
    child = sub.add_parser("_trial", help=argparse.SUPPRESS)
    for p in (run, test, child):
        p.add_argument("--seed", type=int, default=797543261)
    child.add_argument("--case", choices=CASES, required=True)
    child.add_argument("--page", type=int, choices=PAGES, required=True)
    child.add_argument("--layout", choices=LAYOUTS, required=True)
    child.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "self-test":
        self_test(args.seed)
        return 0
    if args.action == "_trial":
        report = dict(
            case=args.case,
            page=args.page,
            layout=args.layout,
            seed=args.seed,
            environment=environment(),
            status="ERROR",
        )
        try:
            with torch.inference_mode():
                trial(args, report)
        except Exception:
            report.update(status="ERROR", error=traceback.format_exc())
            print(report["error"], file=sys.stderr)
        write_json(Path("result.json"), report)
        return 0 if report["status"] == "PASS" else 1
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    if visible and not set(args.devices).issubset(devices_arg(visible)):
        parser.error(f"--devices must be a subset of ASCEND_RT_VISIBLE_DEVICES={visible}")
    folder = args.out.resolve()
    if folder.exists():
        parser.error("--out must be a new directory")
    folder.mkdir(parents=True)
    try:
        info = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        (folder / "npu-smi.txt").write_text(info.stdout + info.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        (folder / "npu-smi.txt").write_text(str(e))
    profile_card = None if args.no_profile else args.devices[0]
    with ThreadPoolExecutor(max_workers=len(args.devices)) as workers:
        jobs = [workers.submit(run_card, args, folder, d, profile_card) for d in args.devices]
        records = [r for job in jobs for r in job.result()]
    return summarize(records, args, folder)


if __name__ == "__main__":
    raise SystemExit(main())
