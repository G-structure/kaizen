#pragma once

#include <vector>
#include "fieldElement.hpp"

// CUDA accelerated operations for zkPoT
namespace cuda_kernels {

/**
 * Performs matrix multiplication C = A * B using CUDA acceleration
 * 
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @return Result matrix C (m x n)
 */
std::vector<std::vector<F>> matrix_multiply(
    const std::vector<std::vector<F>>& A,
    const std::vector<std::vector<F>>& B);

/**
 * Applies ReLU activation function to each element in the input vector
 * 
 * @param input Input vector
 * @return Vector with ReLU applied to each element
 */
std::vector<F> apply_relu(const std::vector<F>& input);

/**
 * Applies average pooling to a flattened tensor
 * 
 * @param input Flattened tensor
 * @param input_dim Dimension of the square input tensor
 * @param kernel_size Size of the pooling kernel
 * @param stride Stride for pooling
 * @return Flattened result of average pooling
 */
std::vector<F> apply_avg_pooling(
    const std::vector<F>& input,
    int input_dim,
    int kernel_size,
    int stride);

/**
 * Applies max pooling to a flattened tensor
 * 
 * @param input Flattened tensor
 * @param input_dim Dimension of the square input tensor
 * @param kernel_size Size of the pooling kernel
 * @param stride Stride for pooling
 * @return Flattened result of max pooling
 */
std::vector<F> apply_max_pooling(
    const std::vector<F>& input,
    int input_dim,
    int kernel_size,
    int stride);

} // namespace cuda_kernels 