#pragma once

#include <vector>
#include "fieldElement.hpp"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Wrapper for CUTLASS functionality to accelerate zkPoT operations
namespace cutlass_wrapper {

// Convert vector<F> (fieldElement) to float array for CUTLASS
float* convert_to_float_array(const std::vector<F>& input);

// Convert float array back to vector<F>
std::vector<F> convert_from_float_array(const float* data, size_t size);

// Matrix multiplication: C = A * B
// A is m x k, B is k x n, C is m x n
std::vector<std::vector<F>> matrix_multiply(
    const std::vector<std::vector<F>>& A,
    const std::vector<std::vector<F>>& B);

// Apply ReLU activation to a vector
std::vector<F> apply_relu(const std::vector<F>& input);

// Apply max pooling to a tensor represented as a flattened vector
// input_dim is the width/height of the square input
// kernel_size is the width/height of the pooling window
// stride is the step size
std::vector<F> apply_max_pooling(
    const std::vector<F>& input,
    int input_dim,
    int kernel_size,
    int stride);

// Apply average pooling to a tensor
std::vector<F> apply_avg_pooling(
    const std::vector<F>& input,
    int input_dim,
    int kernel_size,
    int stride);

} // namespace cutlass_wrapper 