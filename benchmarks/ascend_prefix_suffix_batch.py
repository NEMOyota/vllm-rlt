"""Compare individual, paired and four-way cached suffix attention on the FP32 matrix."""

from dataclasses import dataclass
from functools import partial

from benchmarks.ascend_prefix_scenarios import suites

if __package__:
    from . import ascend_prefix_two_plus_two as two
else:  # Run from an archive alongside the server's locally patched matrix checkout.
    import ascend_prefix_two_plus_two as two


CASES = {c.name: c for c in suites()["full"] if c.name in ("c4_h4", "c2_h1", "c2_h2", "c4_h0")}
WIDTHS = {"original": 1, "width2": 2, "width4": 4}


@dataclass
class SuffixPlan:
    batches: list[two.SuffixBatch]

    @classmethod
    def plan(cls, batch, groups, *, width):
        # No complete prompt is required. Each suffix still reads from its own
        # page table and uses the native absolute-position mask at this depth.
        if not two.equal_suffixes(batch, groups):
            return None
        return cls(
            [
                two.SuffixBatch.from_groups(groups[start : start + width])
                for start in range(0, len(groups), width)
            ]
        )

    def apply(self, native, owner, q, keys, values, tables, flat_rows, result):
        for batch in self.batches:
            result = batch.attend(q, keys, tables, flat_rows, result)
        return result


class SuffixExperiment(two.TwoPlusTwo):
    label = "SUFFIX_BATCH"
    case_name = "c4_h4"
    intermediate_modes = ("width2",)
    candidate = "width4"
    cold_count = 0

    def suffix_width(self, mode, hits):
        return min(WIDTHS[mode], hits)

    def variant_route(self, mode, audit):
        return two.mixed.route(
            mode != "original",
            audit,
            cold_count=self.cold_count,
            batch_size=CASES[self.case_name].concurrency,
            plan_batch=partial(SuffixPlan.plan, width=WIDTHS[mode]),
        )


def main(argv=None):
    parser = two.paired.argument_parser(SuffixExperiment, __doc__, reference_required=True)
    parser.add_argument("--case", choices=sorted(CASES), default="c4_h4")
    parser.add_argument(
        "--width",
        type=int,
        choices=(2, 4),
        help="compare just this maximum suffix batch width against original",
    )
    args = parser.parse_args(argv)
    if args.width:
        widths = (args.width,)
    elif args.case == "c4_h4":
        widths = (2, 4)
    else:
        # Both limits exercise the same path when at most two requests hit.
        widths = (4,)
    modes = ("original", *(f"width{width}" for width in widths))
    return two.paired.run_experiment(
        parser, args, SuffixExperiment, case_name=args.case, modes=modes
    )


if __name__ == "__main__":
    main()
