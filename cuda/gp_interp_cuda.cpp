#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> gp_interp_cuda_forward(
    torch::Tensor input,
    torch::Tensor means,
    torch::Tensor stds,
    int radius);

std::vector<torch::Tensor> gp_interp_cuda_backward(
    torch::Tensor grad_interp_image,
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius,
    torch::Tensor interp_image,
    torch::Tensor pixel_weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gp_interp_forward(
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius) {

  CHECK_INPUT(image);
  CHECK_INPUT(means);
  CHECK_INPUT(stds);

  return gp_interp_cuda_forward(image, means, stds, radius);
}

std::vector<torch::Tensor> gp_interp_backward(
    torch::Tensor grad_interp_image,
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius,
    torch::Tensor interp_image,
    torch::Tensor pixel_weights) {

  CHECK_INPUT(grad_interp_image);
  CHECK_INPUT(image);
  CHECK_INPUT(means);
  CHECK_INPUT(stds);
  CHECK_INPUT(interp_image);
  CHECK_INPUT(pixel_weights);

  return gp_interp_cuda_backward(
      grad_interp_image,
      image,
      means,
      stds,
      radius,
      interp_image,
      pixel_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gp_interp_forward, "gp_interp forward (CUDA)");
  m.def("backward", &gp_interp_backward, "gp_interp backward (CUDA)");
}
