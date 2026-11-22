"""Reporting shared by the native-BF16 engine and HTTP experiments."""

import csv
import hashlib
import json
import struct
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from benchmarks.ascend_main_prefix_combo import environment
from benchmarks.preemption import checkpoint_identity, percentile
from vllm_rlt.device import resolve_device


def ids_digest(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}q", *tokens)).hexdigest()


def quantiles(values):
    return {
        f"p{p}": round(percentile(values, p / 100), 3) if values else None for p in (50, 95, 99)
    }


def savings(left, right):
    pairs = [a - b for a, b in zip(left, right, strict=True)]
    return dict(
        pairs_ms=[round(delta, 3) for delta in pairs],
        median_saved_ms=round(median(pairs), 3),
        median_saved_pct=round(median(100 * d / t for d, t in zip(pairs, left, strict=True)), 2)
        if all(t > 0 for t in left)
        else None,
        faster_pairs=sum(n > 0 for n in pairs),
        pairs=len(pairs),
    )


def write_csv(path, rows):
    if rows:
        with path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class Report:
    def __init__(self, kind, folder):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.folder = folder or Path("results") / f"main-prefix-{kind.lower()}-{stamp}"
        self.folder.mkdir(exist_ok=False, parents=True)
        self.kind = kind
        self.location = {}
        self.stream = (self.folder / "raw.jsonl").open("x", encoding="utf-8")
        print("RESULTS", self.folder.resolve(), flush=True)

    def record(self, tag, *, echo=False, **data):
        line = json.dumps(dict(kind=tag, **data), allow_nan=False)
        print(line, file=self.stream, flush=True)
        if echo:
            print(tag, json.dumps(data, allow_nan=False), flush=True)

    def review(self, status, **data):
        result = dict(status=status, **data)
        (self.folder / "review.json").write_text(json.dumps(result, indent=2) + "\n")
        self.record(self.kind + "_REVIEW", echo=True, **result)
        return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[status]

    def execute(self, args, run):
        try:
            self.location = {"phase": "setup"}
            device = resolve_device("npu:0")
            import torch

            torch.npu.set_device(device)
            info = environment(device)
            info["execution"] = "native BF16, single-stream synchronous ascend_eager, packed SDPA"
            if self.kind == "SERVICE":
                info.update(seed=args.seed, output_tokens=[8, 16])
            self.record(
                "ENV",
                **info,
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                checkpoint=checkpoint_identity(args.model),
            )
            return run(args, device, self)
        except Exception as error:
            traceback.print_exc()
            return self.review(
                "INCOMPLETE", stopped_at=self.location, detail=f"{type(error).__name__}: {error}"
            )
        finally:
            self.stream.close()
