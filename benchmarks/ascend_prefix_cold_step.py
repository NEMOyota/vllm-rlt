"""Step 1. Compare the framework's cold-prompt grouping switch on the accepted FP32 matrix."""

from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

from vllm_rlt.kernels.ascend_eager import AscendEagerAttention

if __package__:
    from . import ascend_prefix_three_plus_one as mixed
else:
    import ascend_prefix_three_plus_one as mixed


class FrameworkCold(mixed.ThreePlusOne):
    label = "COLD_STEP1"
    candidate = "framework_cold"
    profile_pair = None
    stock_methods = (*mixed.ThreePlusOne.stock_methods, "_prefill_attention", "_plan_cold_prefills")

    def execution_options(self, mode):
        return {"ascend_batch_cold_prefill": mode == self.candidate}

    @staticmethod
    def new_audit():
        return dict(mixed.ThreePlusOne.new_audit(), cold_calls=0, suffix_calls=0)

    def run_variant(self, mode, audit, trial_args, trial_kwargs, profile_target=None):
        # Supply the option when the matrix builds its usual fresh StageEngine.
        workload = mixed.paired.workload
        config = partial(workload.ExecutionConfig, **self.execution_options(mode))
        with patch.object(workload, "ExecutionConfig", config):
            return super().run_variant(mode, audit, trial_args, trial_kwargs, profile_target)

    @contextmanager
    def variant_route(self, mode, audit):
        if audit is None:
            yield
            return
        attend = AscendEagerAttention.attend_metadata

        def observe(kernel, *args):
            before = kernel.cold_prefill_calls
            suffix_before = kernel.suffix_prefill_calls
            output = attend(kernel, *args)
            cold = kernel.cold_prefill_calls - before
            suffix = kernel.suffix_prefill_calls - suffix_before
            audit["cold_calls"] += cold
            audit["suffix_calls"] += suffix
            audit["split_calls"] += int(bool(cold or suffix))
            return output

        # Count native execution only. The 3+1 prototype routing hook stays disabled.
        with (
            mixed.route(False, audit),
            patch.object(AscendEagerAttention, "attend_metadata", observe),
        ):
            yield


def main(argv=None):
    parser = mixed.paired.argument_parser(FrameworkCold, __doc__, reference_required=True)
    parser.add_argument("--case", choices=("c4_h1", "c4_h2", "c4_h0", "c2_h1"), default="c4_h1")
    args = parser.parse_args(argv)
    return mixed.paired.run_experiment(parser, args, FrameworkCold, case_name=args.case)


if __name__ == "__main__":
    main()
