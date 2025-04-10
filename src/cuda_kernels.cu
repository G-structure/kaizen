#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Matrix multiplication kernel (C = A * B)
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise addition kernel (C = A + B)
__global__ void matrixAddKernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// ReLU activation kernel
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Wrapper functions to call from C++
extern "C" {

// Matrix multiplication wrapper
void cuda_matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Copy result back to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Matrix addition wrapper
void cuda_matrix_add(float* A, float* B, float* C, int size) {
    float *d_A, *d_B, *d_C;
    size_t bytes = size * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    
    // Copy result back to host
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ReLU wrapper
void cuda_relu(float* data, int size) {
    float *d_data;
    size_t bytes = size * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_data, bytes);
    
    // Copy input data from host to device
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    reluKernel<<<gridSize, blockSize>>>(d_data, size);
    
    // Copy result back to host
    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
}

} // extern "C" 