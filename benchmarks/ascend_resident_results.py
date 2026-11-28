"""Capacity mathematics, input fixtures and same-device paired evidence."""

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from math import ceil
from statistics import mean, median

from benchmarks.ascend_prefix_benefit import Workload, workload_inputs
from benchmarks.ascend_resident_observer import dump_lines
from benchmarks.preemption import percentile

GROUPS = dict(
    A=(False, 4),
    B=(True, 4),
    C=(False, 32),
    D=(True, 6),
    E=(True, 8),
    F=(True, 12),
    G=(True, 15),
    H=(True, 16),
    I=(True, 32),
)
PAIRS = (("B", "A"), ("C", "A"), ("I", "B"), ("I", "C"))
METRICS = (
    "output_tok_s",
    "request_s",
    "inclusive_output_tok_s",
    "queue_p50_ms",
    "queue_p95_ms",
    "execution_wait_p50_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "e2e_p95_ms",
    "resident_mean",
    "resident_peak",
    "decode_mean_batch",
    "peak_active_blocks",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


@dataclass(frozen=True)
class Config:
    num_blocks: int = 5280
    max_num_batched_tokens: int = 1024
    prefill_chunk_size: int = 256
    warmup_requests: int = 4
    target_requests: int = 32
    block_size: int = 16
    input_length: int = 4096
    output_length: int = 128
    shared_prefix_length: int = 3072
    max_loops: int = 4

    def validate(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("all dimensions must be positive")
        if self.block_size != 16 or self.max_loops != 4:
            raise ValueError("Ouro Ascend requires 16-token blocks and four loops")
        if not self.shared_prefix_length < self.input_length:
            raise ValueError("the prefix must leave a private suffix")
        if self.num_blocks < self.full_blocks:
            raise ValueError("a single request exceeds the pool")

    @property
    def full_blocks(self):
        return ceil((self.input_length + self.output_length - 1) / self.block_size) * self.max_loops

    @property
    def prefix_blocks(self):
        return self.shared_prefix_length // self.block_size * self.max_loops

    def arm(self, group):
        enabled, limit = GROUPS[group]
        return {**asdict(self), "cache_enabled": enabled, "resident_limit": limit}

    def theory(self, group, hot):
        # Best possible simultaneous packing using the specified mix. Arrival order
        # and subsequent evictions may lower the observed peak. Ties favor more hot.
        enabled, cap = GROUPS[group]
        cap = min(cap, self.target_requests)
        cold = self.target_requests - hot
        prefix = self.prefix_blocks if enabled else 0
        best = (0, 0, 0)
        for h in range(min(hot, cap) + 1):
            hot_cost = (prefix if h else 0) + h * (self.full_blocks - prefix)
            if hot_cost > self.num_blocks:
                continue
            m = min(cap - h, cold, (self.num_blocks - hot_cost) // self.full_blocks)
            best = max(best, (h + m, h, hot_cost + m * self.full_blocks))
        return dict(theory_peak=best[0], theory_blocks_at_best_mix=best[2])


def hot_count(cfg, case, mixed_hot):
    if case == "hot":
        return cfg.target_requests
    if case == "cold":
        return 0
    if case == "evict":
        cold = cfg.num_blocks // cfg.full_blocks
        if cfg.target_requests <= cold or cold * cfg.full_blocks != cfg.num_blocks:
            raise ValueError("evict requires an exact all-cold pool fill followed by hot requests")
        return cfg.target_requests - cold
    if not 0 <= mixed_hot <= cfg.target_requests:
        raise ValueError("mixed hot count outside cohort")
    return mixed_hot


def fixture(cfg, seed, vocab, n_hot, order):
    w = Workload(
        cfg.input_length, cfg.output_length, cfg.target_requests, 0, cfg.shared_prefix_length, n_hot
    )
    donor, targets, expected = workload_inputs(w, seed, vocab)
    ids = list(range(len(targets)))
    if order == "cold-first":
        ids.sort(key=lambda i: bool(expected[i]))
    if order == "hot-first":
        ids.sort(key=lambda i: not expected[i])
    return (
        donor,
        {f"target-{i}": targets[i] for i in ids},
        {f"target-{i}": bool(expected[i]) for i in ids},
    )


def stats(values):
    if not values:
        return dict.fromkeys(("mean", "p50", "p95", "p99"))
    return dict(
        mean=mean(values),
        p50=median(values),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
    )


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def write_csv(path, records):
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def compare(a, b, category, issues):
    if a["input_sha256"] != b["input_sha256"] or a["request_ids"] != b["request_ids"]:
        issues.append(dict(kind="input_mismatch", comparison=category))
        return
    for rid in a["request_ids"]:
        left, right = a["outputs"][rid], b["outputs"][rid]
        if left == right:
            continue
        i = next(
            (i for i, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right))
        )
        issues.append(
            dict(
                kind="bf16_divergence",
                comparison=category,
                request=rid,
                position_1based=i + 1,
                before=left[i : i + 1],
                after=right[i : i + 1],
            )
        )


def report(folder, jobs):
    """Rebuild from raw results, including expected jobs with absent/failed observations."""
    rows, records, issues = [], {}, []
    for job in jobs:
        path = folder / job["path"] / "result.json"
        data = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ident = {k: job[k] for k in ("case", "order", "group", "round", "seed", "mode", "path")}
        row = {
            **ident,
            **{
                k: job["config"][k]
                for k in (
                    "cache_enabled",
                    "resident_limit",
                    "target_requests",
                    "max_num_batched_tokens",
                    "prefill_chunk_size",
                )
            },
            "physical_pool_blocks": data.get("pool", {}).get("blocks"),
            **data.get("metrics", {}),
            **job["theory"],
            "status": data["status"],
            "error": data.get("error", ""),
        }
        rows.append(row)
        if data.get("request_ids") and set(data["request_ids"]) == set(data.get("outputs", {})):
            records[job["path"]] = (row, data)
        if data["status"] != "OK":
            issues.append({**ident, "kind": data["status"], "error": row["error"]})

    def check(left, right, category):
        a, x = left
        b, y = right
        found = []
        compare(x, y, category, found)
        if found:
            for row in (a, b):
                if row["status"] == "OK":
                    row["status"] = "DIAGNOSTIC"
            issues.extend(
                {
                    **r,
                    "seed": a["seed"],
                    "case": a["case"],
                    "order": a["order"],
                    "left": a["path"],
                    "right": b["path"],
                }
                for r in found
            )

    cohorts, repeats = defaultdict(list), defaultdict(list)
    for row, data in records.values():
        key = row["case"], row["order"], row["seed"]
        cohorts[key + (row["round"],)].append((row, data))
        repeats[key + (row["group"],)].append((row, data))
    for runs in repeats.values():
        for later in runs[1:]:
            check(runs[0], later, "within_group_across_rounds")
    for runs in cohorts.values():
        for left, right in combinations(runs, 2):
            a, b = GROUPS[left[0]["group"]], GROUPS[right[0]["group"]]
            category = (
                "cache_switch"
                if a[1] == b[1]
                else "resident_limit"
                if a[0] == b[0]
                else "cache_and_limit"
            )
            check(left, right, category)

    comparisons = []
    rounds = defaultdict(dict)
    for row in rows:
        rounds[row["case"], row["order"], row["round"]][row["group"]] = row
    for (case, order, rnd), groups in rounds.items():
        for numerator, denominator in PAIRS:
            if numerator not in groups or denominator not in groups:
                continue
            a, b = groups[numerator], groups[denominator]
            eligible = a["mode"] == "performance" and a["status"] == b["status"] == "OK"
            for metric in METRICS:
                x, y = a.get(metric), b.get(metric)
                pct = 100 * (x / y - 1) if x is not None and y else None
                comparisons.append(
                    dict(
                        case=case,
                        order=order,
                        round=rnd,
                        pair=f"{numerator}/{denominator}",
                        metric=metric,
                        numerator=x,
                        denominator=y,
                        change_pct=pct,
                        eligible=eligible,
                        numerator_status=a["status"],
                        denominator_status=b["status"],
                    )
                )
    aggregate = []
    by_pair = defaultdict(list)
    for item in comparisons:
        by_pair[item["case"], item["order"], item["pair"], item["metric"]].append(item)
    for (case, order, pair, metric), items in by_pair.items():
        values = [x["change_pct"] for x in items if x["eligible"] and x["change_pct"] is not None]
        aggregate.append(
            dict(
                case=case,
                order=order,
                pair=pair,
                metric=metric,
                n=len(values),
                planned=len(items),
                median_pct=median(values) if values else None,
                min_pct=min(values, default=None),
                max_pct=max(values, default=None),
            )
        )
    arm_rows = []
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["case"], r["order"], r["group"]].append(r)
    for (case, order, group), runs in by_arm.items():
        for metric in METRICS:
            values = [
                r[metric]
                for r in runs
                if r["mode"] == "performance" and r["status"] == "OK" and r.get(metric) is not None
            ]
            arm_rows.append(
                dict(
                    case=case,
                    order=order,
                    group=group,
                    metric=metric,
                    valid=len(values),
                    planned=len(runs),
                    median=median(values) if values else None,
                    minimum=min(values, default=None),
                    maximum=max(values, default=None),
                )
            )
    write_csv(folder / "group_summary.csv", arm_rows)
    write_csv(folder / "summary.csv", rows)
    write_csv(folder / "paired_rounds.csv", comparisons)
    write_csv(folder / "paired_summary.csv", aggregate)
    dump_lines(folder / "exceptions.jsonl", issues)
    lines = [
        "# Fixed-pool resident capacity",
        "",
        "Theoretical limits assume retained, full prefix hits. Times are host-visible. "
        "Only complete, output-matched performance pairs enter the aggregates.",
        "",
        "|Case/order|Group/round|Theory peak|Measured peak|Output tok/s|Queue P95 ms|"
        "TTFT P95 ms|Decode mean batch|Peak allocated GiB|Status|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def cell(x):
        return "—" if x is None else f"{x:.3f}"

    for r in rows:
        gib = r.get("peak_allocated_bytes")
        metrics = [
            cell(r.get(k))
            for k in (
                "resident_peak",
                "output_tok_s",
                "queue_p95_ms",
                "ttft_p95_ms",
                "decode_mean_batch",
            )
        ]
        lines.append(
            f"|{r['case']}/{r['order']}|{r['group']}/{r['round']}|{r['theory_peak']}|"
            + "|".join(metrics)
            + f"|{cell(gib / 2**30 if gib is not None else None)}|{r['status']}|"
        )
    lines.extend(
        [
            "",
            "|Case/order|Output throughput pair|Valid/planned|Median %|Min %|Max %|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in aggregate:
        if r["metric"] == "output_tok_s":
            lines.append(
                f"|{r['case']}/{r['order']}|{r['pair']}|{r['n']}/{r['planned']}|"
                + "|".join(cell(r[k]) for k in ("median_pct", "min_pct", "max_pct"))
                + "|"
            )
    lines.extend(
        [
            "",
            "Inspect steps.jsonl for decode shape and dispatch, residency.jsonl for "
            "queue and physical occupancy, and exceptions.jsonl for first differing tokens "
            "or missing runs. A capacity shortfall is reported separately from errors.",
        ]
    )
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    return rows
