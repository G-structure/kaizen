#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Print CUDA device info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", 
               static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    }
    
    // Test vector addition
    const int arraySize = 1000000;
    const int blockSize = 256;
    const int gridSize = (arraySize + blockSize - 1) / blockSize;
    
    // Allocate host memory
    float *h_a = new float[arraySize];
    float *h_b = new float[arraySize];
    float *h_c = new float[arraySize];
    
    // Initialize host arrays
    for (int i = 0; i < arraySize; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, arraySize * sizeof(float));
    cudaMalloc(&d_b, arraySize * sizeof(float));
    cudaMalloc(&d_c, arraySize * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, arraySize);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result
    bool success = true;
    for (int i = 0; i < arraySize; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Verification failed at index %d: %f != %f + %f\n", 
                   i, h_c[i], h_a[i], h_b[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("GPU vector addition test completed successfully!\n");
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
} 