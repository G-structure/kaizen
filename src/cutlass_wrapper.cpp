#include "cutlass_wrapper.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// Use CUTLASS for fast matrix multiplications
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

namespace cutlass_wrapper {

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Convert vector<F> to float array for CUTLASS
float* convert_to_float_array(const std::vector<F>& input) {
    float* result = new float[input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        // Convert fieldElement to float
        result[i] = static_cast<float>(input[i].get_ulong());
    }
    return result;
}

// Convert float array back to vector<F>
std::vector<F> convert_from_float_array(const float* data, size_t size) {
    std::vector<F> result(size);
    for (size_t i = 0; i < size; i++) {
        // Convert float back to fieldElement
        result[i] = F(static_cast<unsigned long>(data[i]));
    }
    return result;
}

// Matrix multiplication implementation using CUTLASS
std::vector<std::vector<F>> matrix_multiply(
    const std::vector<std::vector<F>>& A,
    const std::vector<std::vector<F>>& B) {
    
    // Get matrix dimensions
    int m = A.size();
    int k = A[0].size();
    int n = B[0].size();
    
    // Validate dimensions
    if (B.size() != k) {
        std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Prepare input data in row-major format
    float* A_data = new float[m * k];
    float* B_data = new float[k * n];
    float* C_data = new float[m * n];
    
    // Initialize C to zeros
    memset(C_data, 0, m * n * sizeof(float));
    
    // Convert A to row-major float array
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A_data[i * k + j] = static_cast<float>(A[i][j].get_ulong());
        }
    }
    
    // Convert B to row-major float array
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_data[i * n + j] = static_cast<float>(B[i][j].get_ulong());
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc(&d_A, m * k * sizeof(float)), "allocating device memory for A");
    check_cuda_error(cudaMalloc(&d_B, k * n * sizeof(float)), "allocating device memory for B");
    check_cuda_error(cudaMalloc(&d_C, m * n * sizeof(float)), "allocating device memory for C");
    
    // Copy data to device
    check_cuda_error(cudaMemcpy(d_A, A_data, m * k * sizeof(float), cudaMemcpyHostToDevice), "copying A to device");
    check_cuda_error(cudaMemcpy(d_B, B_data, k * n * sizeof(float), cudaMemcpyHostToDevice), "copying B to device");
    
    // Define CUTLASS GEMM type
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    
    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80
    >;
    
    // Configure GEMM parameters
    CutlassGemm::Arguments args(
        {m, n, k},                            // problem size (m, n, k)
        {d_A, k},                             // tensor A (ptr, ldm)
        {d_B, n},                             // tensor B (ptr, ldm)
        {d_C, n},                             // tensor C (ptr, ldm)
        {d_C, n},                             // tensor D (ptr, ldm)
        {1.0f, 0.0f}                          // alpha, beta
    );
    
    // Initialize GEMM operator
    CutlassGemm gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    
    if (workspace_size) {
        check_cuda_error(cudaMalloc(&workspace, workspace_size), "allocating GEMM workspace");
    }
    
    // Run GEMM
    cutlass::Status status = gemm_op(args, workspace);
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed with status " << static_cast<int>(status) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Copy result back to host
    check_cuda_error(cudaMemcpy(C_data, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost), "copying result to host");
    
    // Convert result back to vector<vector<F>>
    std::vector<std::vector<F>> C(m, std::vector<F>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = F(static_cast<unsigned long>(C_data[i * n + j]));
        }
    }
    
    // Free memory
    delete[] A_data;
    delete[] B_data;
    delete[] C_data;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (workspace) cudaFree(workspace);
    
    return C;
}

// Simple CUDA kernel for ReLU activation
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Apply ReLU activation to a vector using CUDA
std::vector<F> apply_relu(const std::vector<F>& input) {
    size_t size = input.size();
    
    // Convert input to float array
    float* h_input = convert_to_float_array(input);
    float* h_output = new float[size];
    
    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, size * sizeof(float)), "allocating device memory for ReLU input");
    check_cuda_error(cudaMalloc(&d_output, size * sizeof(float)), "allocating device memory for ReLU output");
    
    // Copy input to device
    check_cuda_error(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice), "copying ReLU input to device");
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, size);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError(), "launching ReLU kernel");
    check_cuda_error(cudaDeviceSynchronize(), "synchronizing ReLU kernel");
    
    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost), "copying ReLU result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result = convert_from_float_array(h_output, size);
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// CUDA kernel for max pooling
__global__ void max_pooling_kernel(float* input, float* output, 
                                  int input_dim, int output_dim,
                                  int kernel_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x < output_dim && out_y < output_dim) {
        int out_idx = out_y * output_dim + out_x;
        float max_val = -INFINITY;
        
        // Compute max value in the kernel window
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = out_x * stride + kx;
                int in_y = out_y * stride + ky;
                
                if (in_x < input_dim && in_y < input_dim) {
                    int in_idx = in_y * input_dim + in_x;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        output[out_idx] = max_val;
    }
}

// Apply max pooling to a tensor
std::vector<F> apply_max_pooling(const std::vector<F>& input, int input_dim, int kernel_size, int stride) {
    int output_dim = (input_dim - kernel_size) / stride + 1;
    size_t input_size = input.size();
    size_t output_size = output_dim * output_dim;
    
    // Check that input is a square tensor
    if (input_size != input_dim * input_dim) {
        std::cerr << "Input size doesn't match specified dimensions" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Convert input to float array
    float* h_input = convert_to_float_array(input);
    float* h_output = new float[output_size];
    
    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, input_size * sizeof(float)), "allocating device memory for pooling input");
    check_cuda_error(cudaMalloc(&d_output, output_size * sizeof(float)), "allocating device memory for pooling output");
    
    // Copy input to device
    check_cuda_error(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice), "copying pooling input to device");
    
    // Launch kernel
    dim3 threads_per_block(16, 16);
    dim3 blocks((output_dim + threads_per_block.x - 1) / threads_per_block.x,
                (output_dim + threads_per_block.y - 1) / threads_per_block.y);
    
    max_pooling_kernel<<<blocks, threads_per_block>>>(d_input, d_output, input_dim, output_dim, kernel_size, stride);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError(), "launching max pooling kernel");
    check_cuda_error(cudaDeviceSynchronize(), "synchronizing max pooling kernel");
    
    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost), "copying pooling result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result(output_size);
    for (size_t i = 0; i < output_size; i++) {
        result[i] = F(static_cast<unsigned long>(h_output[i]));
    }
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

// CUDA kernel for average pooling
__global__ void avg_pooling_kernel(float* input, float* output, 
                                  int input_dim, int output_dim,
                                  int kernel_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x < output_dim && out_y < output_dim) {
        int out_idx = out_y * output_dim + out_x;
        float sum = 0.0f;
        int count = 0;
        
        // Compute average value in the kernel window
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = out_x * stride + kx;
                int in_y = out_y * stride + ky;
                
                if (in_x < input_dim && in_y < input_dim) {
                    int in_idx = in_y * input_dim + in_x;
                    sum += input[in_idx];
                    count++;
                }
            }
        }
        
        output[out_idx] = (count > 0) ? (sum / count) : 0.0f;
    }
}

// Apply average pooling to a tensor
std::vector<F> apply_avg_pooling(const std::vector<F>& input, int input_dim, int kernel_size, int stride) {
    int output_dim = (input_dim - kernel_size) / stride + 1;
    size_t input_size = input.size();
    size_t output_size = output_dim * output_dim;
    
    // Check that input is a square tensor
    if (input_size != input_dim * input_dim) {
        std::cerr << "Input size doesn't match specified dimensions" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Convert input to float array
    float* h_input = convert_to_float_array(input);
    float* h_output = new float[output_size];
    
    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, input_size * sizeof(float)), "allocating device memory for pooling input");
    check_cuda_error(cudaMalloc(&d_output, output_size * sizeof(float)), "allocating device memory for pooling output");
    
    // Copy input to device
    check_cuda_error(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice), "copying pooling input to device");
    
    // Launch kernel
    dim3 threads_per_block(16, 16);
    dim3 blocks((output_dim + threads_per_block.x - 1) / threads_per_block.x,
                (output_dim + threads_per_block.y - 1) / threads_per_block.y);
    
    avg_pooling_kernel<<<blocks, threads_per_block>>>(d_input, d_output, input_dim, output_dim, kernel_size, stride);
    
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError(), "launching avg pooling kernel");
    check_cuda_error(cudaDeviceSynchronize(), "synchronizing avg pooling kernel");
    
    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost), "copying pooling result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result(output_size);
    for (size_t i = 0; i < output_size; i++) {
        result[i] = F(static_cast<unsigned long>(h_output[i]));
    }
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

} // namespace cutlass_wrapper 