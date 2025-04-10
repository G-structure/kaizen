#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiplication (C = A * B)
void cuda_matrix_multiply(float* A, float* B, float* C, int M, int N, int K);

// Matrix addition (C = A + B)
void cuda_matrix_add(float* A, float* B, float* C, int size);

// ReLU activation
void cuda_relu(float* data, int size);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H 