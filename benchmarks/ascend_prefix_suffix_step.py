"""Step 2. Measure the incremental effect of framework suffix batching in c4_h2."""

if __package__:
    from . import ascend_prefix_cold_step as cold
else:
    import ascend_prefix_cold_step as cold

paired = cold.mixed.paired


class FrameworkSuffix(cold.FrameworkCold):
    case_name = "c4_h2"
    label = "SUFFIX_STEP2"
    intermediate_modes = ("framework_cold",)
    candidate = "framework_both"
    batched_suffix_mode = candidate
    stock_methods = (*cold.FrameworkCold.stock_methods, "_plan_suffix_prefills")

    def execution_options(self, mode):
        return {
            "ascend_batch_cold_prefill": mode != "original",
            "ascend_batch_cached_suffix": mode == self.candidate,
        }

    def audit_issues(self, audits, calls):
        issues = super().audit_issues(audits, calls)
        for mode, arms in audits.items():
            for arm, audit in arms.items():
                expected = {
                    "cold_calls": calls if arm == "reuse" and mode != "original" else 0,
                    "suffix_calls": calls if arm == "reuse" and mode == self.candidate else 0,
                }
                for name, count in expected.items():
                    if audit[name] != count:
                        issues.append(
                            dict(mode=mode, arm=arm, check=name, expected=count, actual=audit[name])
                        )
        return issues

    def timing_summary(self):
        return paired.timing_comparisons(self.runs, self.modes)


def main(argv=None):
    parser = paired.argument_parser(FrameworkSuffix, __doc__, reference_required=True)
    return paired.run_experiment(parser, parser.parse_args(argv), FrameworkSuffix)


if __name__ == "__main__":
    main()
