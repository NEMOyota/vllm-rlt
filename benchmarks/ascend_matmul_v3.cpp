// Single-operator bridge to the MatMulV3 shipped with CANN.
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
at::Tensor projection(const at::Tensor& x, const at::Tensor& weight) {
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1),
                "expected packed X[M,K] and weight[N,K]");
    TORCH_CHECK(x.device() == weight.device() && x.scalar_type() == at::kBFloat16 &&
                weight.scalar_type() == at::kBFloat16, "expected BF16 X and weight on one NPU");
    auto y = at::empty({x.size(0), weight.size(0)}, x.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("MatMulV3")
        .Input(x, "x1", ACL_FORMAT_ND)
        .Input(weight, "x2", ACL_FORMAT_ND)
        .Output(y, "y", ACL_FORMAT_ND)
        .Attr("transpose_x1", false)
        .Attr("transpose_x2", true)
        .Run();
    return y;
}
} // namespace

TORCH_LIBRARY(rlt_v3_bench, m) {
    m.def("projection(Tensor x, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(rlt_v3_bench, PrivateUse1, m) {
    m.impl("projection", projection);
}
