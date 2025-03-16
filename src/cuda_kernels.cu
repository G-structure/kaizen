#include "cuda_kernels.h"
#include "fieldElement.hpp"  // For fieldElement definition
#include "config_pc.hpp"     // For F type definition
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// Using namespace to access F type
using namespace std;
// Using virgo namespace for fieldElement
using namespace virgo;

// Helper function to check CUDA errors
static void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Convert vector of fieldElements to float array for CUDA
static float* convertToFloatArray(const std::vector<F>& input) {
    float* result = new float[input.size()];
    for (size_t i = 0; i < input.size(); i++) {
        // Use real component of F (unsigned long long real)
        result[i] = static_cast<float>(input[i].real);
    }
    return result;
}

// Convert float array back to vector of fieldElements
static std::vector<F> convertFromFloatArray(const float* data, size_t size) {
    std::vector<F> result(size);
    for (size_t i = 0; i < size; i++) {
        // Create F using constructor that takes long long
        result[i] = F(static_cast<long long>(data[i]));
    }
    return result;
}

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for average pooling
__global__ void avgPoolingKernel(const float* input, float* output, 
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

// CUDA kernel for max pooling
__global__ void maxPoolingKernel(const float* input, float* output, 
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

namespace cuda_kernels {

std::vector<std::vector<F>> matrix_multiply(
    const std::vector<std::vector<F>>& A,
    const std::vector<std::vector<F>>& B) {
    
    int m = A.size();
    int k = A[0].size();
    int n = B[0].size();
    
    // Validate dimensions
    if (B.size() != k) {
        std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Prepare input data in row-major format
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];
    
    // Convert A to row-major float array - using real member instead of get_ulong()
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = static_cast<float>(A[i][j].real);
        }
    }
    
    // Convert B to row-major float array - using real member instead of get_ulong()
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i * n + j] = static_cast<float>(B[i][j].real);
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, m * k * sizeof(float)), "allocating device memory for A");
    checkCudaError(cudaMalloc(&d_B, k * n * sizeof(float)), "allocating device memory for B");
    checkCudaError(cudaMalloc(&d_C, m * n * sizeof(float)), "allocating device memory for C");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice), 
                   "copying A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice), 
                   "copying B to device");
    
    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    
    // Check for errors
    checkCudaError(cudaGetLastError(), "launching matrix multiply kernel");
    checkCudaError(cudaDeviceSynchronize(), "synchronizing matrix multiply kernel");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost), 
                   "copying result to host");
    
    // Convert result back to vector<vector<F>>
    std::vector<std::vector<F>> C(m, std::vector<F>(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = F(static_cast<long long>(h_C[i * n + j]));
        }
    }
    
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

std::vector<F> apply_relu(const std::vector<F>& input) {
    size_t size = input.size();
    
    // Convert input to float array
    float* h_input = convertToFloatArray(input);
    float* h_output = new float[size];
    
    // Allocate device memory
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "allocating device memory for ReLU input");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "allocating device memory for ReLU output");
    
    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice), 
                   "copying ReLU input to device");
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, size);
    
    // Check for errors
    checkCudaError(cudaGetLastError(), "launching ReLU kernel");
    checkCudaError(cudaDeviceSynchronize(), "synchronizing ReLU kernel");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost), 
                   "copying ReLU result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result = convertFromFloatArray(h_output, size);
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

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
    float* h_input = convertToFloatArray(input);
    float* h_output = new float[output_size];
    
    // Allocate device memory
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size * sizeof(float)), 
                   "allocating device memory for pooling input");
    checkCudaError(cudaMalloc(&d_output, output_size * sizeof(float)), 
                   "allocating device memory for pooling output");
    
    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice), 
                   "copying pooling input to device");
    
    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    avgPoolingKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, input_dim, output_dim, kernel_size, stride);
    
    // Check for errors
    checkCudaError(cudaGetLastError(), "launching avg pooling kernel");
    checkCudaError(cudaDeviceSynchronize(), "synchronizing avg pooling kernel");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost), 
                   "copying pooling result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result(output_size);
    for (size_t i = 0; i < output_size; i++) {
        result[i] = F(static_cast<long long>(h_output[i]));
    }
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

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
    float* h_input = convertToFloatArray(input);
    float* h_output = new float[output_size];
    
    // Allocate device memory
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size * sizeof(float)), 
                   "allocating device memory for pooling input");
    checkCudaError(cudaMalloc(&d_output, output_size * sizeof(float)), 
                   "allocating device memory for pooling output");
    
    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice), 
                   "copying pooling input to device");
    
    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    maxPoolingKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, input_dim, output_dim, kernel_size, stride);
    
    // Check for errors
    checkCudaError(cudaGetLastError(), "launching max pooling kernel");
    checkCudaError(cudaDeviceSynchronize(), "synchronizing max pooling kernel");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost), 
                   "copying pooling result to host");
    
    // Convert result back to vector<F>
    std::vector<F> result(output_size);
    for (size_t i = 0; i < output_size; i++) {
        result[i] = F(static_cast<long long>(h_output[i]));
    }
    
    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

} // namespace cuda_kernels 