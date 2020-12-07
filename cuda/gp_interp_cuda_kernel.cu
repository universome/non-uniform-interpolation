#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_gaussian_density(
    scalar_t x, scalar_t y,
    scalar_t mean_x, scalar_t mean_y,
    scalar_t std_x, scalar_t std_y) {

    const auto x_exp_term = (x - mean_x) / std_x;
    const auto y_exp_term = (y - mean_y) / std_y;

    const auto exp_term = __expf(-0.5 * (x_exp_term * x_exp_term + y_exp_term * y_exp_term));
    const auto std_term = 1.0 / (std_x * std_y);
    const auto const_term = 1.0 / (2.0 * 3.141592654);

    return exp_term * std_term * const_term;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_distance_weight(
    scalar_t x, scalar_t y,
    scalar_t mean_x, scalar_t mean_y) {

    const auto x_diff = (x - mean_x);
    const auto y_diff = (y - mean_y);
    const auto dist = x_diff * x_diff + y_diff * y_diff;
    const auto weight = 1 / dist;

    return weight;
}

template <typename scalar_t>
__device__ scalar_t clamp(scalar_t x, scalar_t min_val, scalar_t max_val)
{
  return max(min_val, min(max_val, x));
}



template <typename scalar_t>
__global__ void gp_interp_compute_color_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> means,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stds,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pixel_weights) {

    const int point_idx = threadIdx.x;
    const int radius = blockDim.x / 2;
    const unsigned int center_x = static_cast<int>(clamp(round(means[point_idx][0]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(2))));
    const unsigned int center_y = static_cast<int>(clamp(round(means[point_idx][1]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(1))));
    const unsigned int shift_x = threadIdx.x - radius;
    const unsigned int shift_y = threadIdx.y - radius;
    scalar_t pixel_pos_x = static_cast<scalar_t>(center_x + shift_x);
    scalar_t pixel_pos_y = static_cast<scalar_t>(center_y + shift_y);

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < image.size(2) && pixel_pos_y < image.size(1)) {
        // scalar_t weight = compute_gaussian_density(
        //     pixel_pos_x,
        //     pixel_pos_y,
        //     means[point_idx][0],
        //     means[point_idx][1],
        //     stds[point_idx][0],
        //     stds[point_idx][1]);
        scalar_t weight = compute_distance_weight(
            pixel_pos_x,
            pixel_pos_y,
            means[point_idx][0],
            means[point_idx][1]);

        if (weight > 0.0) {
            for (int c = 0; c < image.size(0); c++) {
                // const scalar_t color_value = compute_color_bilinear(pixel_pos_x, pixel_pos_y, c, image);
                const scalar_t color_value = image[c][center_y][center_x];
                atomicAdd(&pixel_weights[pixel_pos_y][pixel_pos_x], weight);
                atomicAdd(&output_image[c][pixel_pos_y][pixel_pos_x], color_value * weight);
            }
        }
    }
}


template <typename scalar_t>
__global__ void gp_interp_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output_image,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> means,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stds,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pixel_weights,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_means,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_stds) {

    //...
}
} // namespace

std::vector<torch::Tensor> gp_interp_cuda_forward(
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius) {

    const auto num_points = means.size(0);
    auto output_image = torch::zeros_like(image);
    auto pixel_weights = torch::zeros({image.size(1), image.size(2)}).to(output_image.device());

    const dim3 threads(radius * 2, radius * 2);
    const int blocks = num_points;

    AT_DISPATCH_FLOATING_TYPES(image.type(), "gp_interp_compute_color_kernel", ([&] {
    gp_interp_compute_color_kernel<scalar_t><<<blocks, threads>>>(
        image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        means.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        stds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        pixel_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    // AT_DISPATCH_FLOATING_TYPES(image.type(), "gp_interp_normalize_color_kernel", ([&] {
    // gp_interp_normalize_color_kernel<scalar_t><<<blocks, threads>>>(
    //     image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
    //     means.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //     stds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    //     output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
    //     pixel_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    // }));

    return {output_image, pixel_weights};
}


std::vector<torch::Tensor> gp_interp_cuda_backward(
    torch::Tensor grad_output_image,
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius,
    torch::Tensor output_image,
    torch::Tensor pixel_weights) {

    auto grad_image = torch::zeros_like(image);
    auto grad_means = torch::zeros_like(means);
    auto grad_stds = torch::zeros_like(stds);

    const dim3 threads(radius * 2, radius * 2);
    const auto num_points = means.size(0);
    const int blocks = num_points;

    AT_DISPATCH_FLOATING_TYPES(image.type(), "gp_interp_cuda_backward", ([&] {
    gp_interp_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        means.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        stds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        pixel_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        grad_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        grad_means.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        grad_stds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return {grad_image, grad_means, grad_stds};
}
