

#include <torch/extension.h>

#include <iostream>

torch::Tensor mondrian_integral_op_cuda(torch::Tensor poly);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

torch::Tensor mondrian_integral_op(
	torch::Tensor batch_idx,
        torch::Tensor v,
	torch::Tensor src_coords,
	torch::Tensor tgt_coords,
	torch::Tensor cluster,
	) {
        CHECK_INPUT(poly);
        return ibf_minmax_cuda(poly);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ibf_minmax",
 	&ibf_minmax,
	"Computes the min and max values of a Bernstein polynomial in implifict form");
}
