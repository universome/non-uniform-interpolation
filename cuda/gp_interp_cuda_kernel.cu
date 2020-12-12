#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
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
__device__ __forceinline__ scalar_t d_normal_pdf_d_mu_i(
    scalar_t x, scalar_t mean_x, scalar_t std_x, scalar_t density_value) {

    return density_value * (x - mean_x) / (std_x * std_x);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t d_normal_pdf_d_std_i(
    scalar_t x, scalar_t mean_x, scalar_t std_x, scalar_t density_value) {
    const auto d_denom_term = -density_value / std_x;
    const auto d_exp_term = density_value * (x - mean_x) * (x - mean_x) / (std_x * std_x * std_x);

    return d_denom_term + d_exp_term;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_distance_weight(
    scalar_t x, scalar_t y,
    scalar_t mean_x, scalar_t mean_y) {

    const auto x_diff = (x - mean_x);
    const auto y_diff = (y - mean_y);
    const auto dist2 = x_diff * x_diff + y_diff * y_diff;
    const auto weight = 1.0 / dist2;

    return weight;
}


template <typename scalar_t>
__device__ scalar_t clamp(scalar_t x, scalar_t min_val, scalar_t max_val) {
    return max(min_val, min(max_val, x));
}


template <typename scalar_t>
__device__ scalar_t compute_color_bilinear(
    scalar_t x,
    scalar_t y,
    const int c,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image,
    bool use_d_x = false,
    bool use_d_y = false) {
    // Selects a color from an image using bilinear interpolation

    // i, j = math.floor(y.item()), math.floor(x.item())
    const int j = min(max(static_cast<int>(floor(x)), 0), image.size(2));
    const int i = min(max(static_cast<int>(floor(y)), 0), image.size(1));

    scalar_t c0 = image[c][i][j];
    scalar_t c1 = image[c][i][min(j + 1, image.size(2) - 1)];
    scalar_t c2 = image[c][min(i + 1, image.size(1) - 1)][min(j + 1, image.size(2) - 1)];
    scalar_t c3 = image[c][min(i + 1, image.size(1) - 1)][j];

    // Computing the derivative here is relatively easy
    // Since it is a linear function wrt to mu_x and mu_y,
    // We just replace them with ones in those places where they are used
    scalar_t w_top = use_d_y ? 1.0 : (y - static_cast<scalar_t>(i));
    scalar_t w_left = use_d_x ? 1.0 : (x - static_cast<scalar_t>(j));

    // Step 1: interpolate along x-axis
    scalar_t color_top = c0 * (1.0 - w_left) + c1 * w_left;
    scalar_t color_bot = c3 * (1.0 - w_left) + c2 * w_left;

    // Step 2: interpolate along y-axis
    scalar_t color = color_top * (1.0 - w_top) + w_top * color_bot;

    return color;
}



template <typename scalar_t>
__global__ void gp_interp_compute_color_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> means,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stds,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pixel_weights) {

    const int point_idx = blockDim.y * blockIdx.y + blockIdx.x;
    const int radius = blockDim.x / 2;
    const int center_x = static_cast<int>(clamp(round(means[point_idx][0]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(2))));
    const int center_y = static_cast<int>(clamp(round(means[point_idx][1]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(1))));
    const int shift_x = threadIdx.x - radius;
    const int shift_y = threadIdx.y - radius;
    const int pixel_pos_x = center_x + shift_x;
    const int pixel_pos_y = center_y + shift_y;

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < image.size(2) && pixel_pos_y < image.size(1)) {
        scalar_t weight = compute_gaussian_density(
            static_cast<scalar_t>(pixel_pos_x),
            static_cast<scalar_t>(pixel_pos_y),
            means[point_idx][0],
            means[point_idx][1],
            stds[point_idx][0],
            stds[point_idx][1]);
        // scalar_t weight = compute_distance_weight(
        //     static_cast<scalar_t>(pixel_pos_x),
        //     static_cast<scalar_t>(pixel_pos_y),
        //     means[point_idx][0],
        //     means[point_idx][1]);
        // scalar_t weight = means[point_idx][0] + means[point_idx][1];

        if (weight > 0.0) {
            for (int c = 0; c < image.size(0); c++) {
                // const scalar_t color_value = image[c][center_y][center_x]; // TODO: keep the color in the shared memory
                const scalar_t color_value = compute_color_bilinear(means[point_idx][0], means[point_idx][1], c, image);
                atomicAdd(&output_image[c][pixel_pos_y][pixel_pos_x], color_value * weight);
            }

            atomicAdd(&pixel_weights[pixel_pos_y][pixel_pos_x], weight);
        }
    }
}


template <typename scalar_t>
__global__ void gp_interp_normalize_color_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_image,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pixel_weights) {

    const int pixel_pos_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_pos_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < output_image.size(2) && pixel_pos_y < output_image.size(1)) {
        const scalar_t weight = pixel_weights[pixel_pos_y][pixel_pos_x];

        if (weight > 0.0) {
            for (int c = 0; c < output_image.size(0); c++) {
                output_image[c][pixel_pos_y][pixel_pos_x] /= weight;
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

    const int point_idx = blockDim.y * blockIdx.y + blockIdx.x;
    const int radius = blockDim.x / 2;
    const int center_x = static_cast<int>(clamp(round(means[point_idx][0]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(2))));
    const int center_y = static_cast<int>(clamp(round(means[point_idx][1]), static_cast<scalar_t>(0.0), static_cast<scalar_t>(image.size(1))));
    const int shift_x = threadIdx.x - radius;
    const int shift_y = threadIdx.y - radius;
    const int pixel_pos_x = center_x + shift_x;
    const int pixel_pos_y = center_y + shift_y;

    if (pixel_pos_x >= 0 && pixel_pos_y >= 0 && pixel_pos_x < image.size(2) && pixel_pos_y < image.size(1)) {
        scalar_t point_density = compute_gaussian_density(
            static_cast<scalar_t>(pixel_pos_x),
            static_cast<scalar_t>(pixel_pos_y),
            means[point_idx][0],
            means[point_idx][1],
            stds[point_idx][0],
            stds[point_idx][1]);
        scalar_t total_weight = pixel_weights[pixel_pos_y][pixel_pos_x];

        if (point_density > 0.0) {
            scalar_t d_v_d_mu_x = 0.0;
            scalar_t d_v_d_mu_y = 0.0;
            scalar_t d_v_d_std_x = 0.0;
            scalar_t d_v_d_std_y = 0.0;

            scalar_t common_multiplier = (1.0 - point_density / total_weight) / total_weight;
            // scalar_t common_multiplier = 1.0 / total_weight;
            scalar_t d_weight_d_mu_x = common_multiplier * d_normal_pdf_d_mu_i(
                static_cast<scalar_t>(pixel_pos_x), means[point_idx][0], stds[point_idx][0], point_density);
            scalar_t d_weight_d_mu_y = common_multiplier * d_normal_pdf_d_mu_i(
                static_cast<scalar_t>(pixel_pos_y), means[point_idx][1], stds[point_idx][1], point_density);
            scalar_t d_weight_d_std_x = common_multiplier * d_normal_pdf_d_std_i(
                static_cast<scalar_t>(pixel_pos_x), means[point_idx][0], stds[point_idx][0], point_density);
            scalar_t d_weight_d_std_y = common_multiplier * d_normal_pdf_d_std_i(
                static_cast<scalar_t>(pixel_pos_y), means[point_idx][1], stds[point_idx][1], point_density);

            for (int c = 0; c < image.size(0); c++) {
                const scalar_t d_loss_d_pixel_color = grad_output_image[c][pixel_pos_y][pixel_pos_x];
                // const scalar_t color_value = image[c][center_y][center_x]; // TODO: keep the color in the shared memory?
                const scalar_t color_value = compute_color_bilinear(means[point_idx][0], means[point_idx][1], c, image);
                const scalar_t d_color_d_mu_x = compute_color_bilinear(means[point_idx][0], means[point_idx][1], c, image, true, false);
                const scalar_t d_color_d_mu_y = compute_color_bilinear(means[point_idx][0], means[point_idx][1], c, image, false, true);

                d_v_d_mu_x += d_loss_d_pixel_color * (d_color_d_mu_x * point_density / total_weight + color_value * d_weight_d_mu_x);
                d_v_d_mu_y += d_loss_d_pixel_color * (d_color_d_mu_y * point_density / total_weight + color_value * d_weight_d_mu_y);
                d_v_d_std_x += d_loss_d_pixel_color * color_value * d_weight_d_std_x;
                d_v_d_std_y += d_loss_d_pixel_color * color_value * d_weight_d_std_y;
            }

            atomicAdd(&grad_means[point_idx][0], d_v_d_mu_x);
            atomicAdd(&grad_means[point_idx][1], d_v_d_mu_y);
            atomicAdd(&grad_stds[point_idx][0], d_v_d_std_x);
            atomicAdd(&grad_stds[point_idx][1], d_v_d_std_y);
        }
    }
}
} // namespace

std::vector<torch::Tensor> gp_interp_cuda_forward(
    torch::Tensor image,
    torch::Tensor means,
    torch::Tensor stds,
    int radius) {

    const auto num_points = means.size(0);
    auto output_image = torch::zeros_like(image).contiguous();
    auto pixel_weights = torch::zeros({image.size(1), image.size(2)}).to(output_image.device()).contiguous();

    {
        const dim3 threads(radius * 2 + 1, radius * 2 + 1);
        const int blocks = num_points;

        AT_DISPATCH_FLOATING_TYPES(image.type(), "gp_interp_compute_color_kernel", ([&] {
        gp_interp_compute_color_kernel<scalar_t><<<blocks, threads>>>(
            image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            means.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            stds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            pixel_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    {
        const dim3 threads(16, 16);
        const dim3 blocks((image.size(2) + 16 - 1) / 16, (image.size(1) + 16 - 1) / 16);
        AT_DISPATCH_FLOATING_TYPES(image.type(), "gp_interp_normalize_color_kernel", ([&] {
        gp_interp_normalize_color_kernel<scalar_t><<<blocks, threads>>>(
            output_image.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            pixel_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

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

    auto grad_image = torch::zeros_like(image).contiguous();
    auto grad_means = torch::zeros_like(means).contiguous();
    auto grad_stds = torch::zeros_like(stds).contiguous();

    const dim3 threads(radius * 2 + 1, radius * 2 + 1);
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
    AT_CUDA_CHECK(cudaGetLastError());

    return {grad_image, grad_means, grad_stds};
}
