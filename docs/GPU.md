# Kaizen with GPU Acceleration

This document explains how to run Kaizen with GPU acceleration on ARM architecture.

## Prerequisites

1. NVIDIA GPU hardware
2. Docker with NVIDIA Container Toolkit installed
3. ARM-based machine (e.g., Apple M1/M2/M3)

## Quick Start

The easiest way to run Kaizen with GPU acceleration is to use our Docker-based approach:

```bash
./run_gpu.sh
```

This script will:
1. Build a Docker container with CUDA support
2. Give you options to run different ML model tests with GPU acceleration
3. Allow you to verify GPU functionality

## GPU Implementation Details

We've added GPU acceleration to the most compute-intensive parts of the codebase:

1. **Matrix Operations**: Matrix multiplication and addition operations are now GPU-accelerated using CUDA kernels.
2. **Activation Functions**: ReLU and other activation functions use the GPU when available.
3. **Hash Functions**: SHA-256 implementation optimized for GPU.

## Running Individual Tests

You can run specific tests directly:

- GPU functionality test: `docker run --gpus all -it kaizen-gpu ./test_gpu`
- AlexNet test: `docker run --gpus all -it kaizen-gpu ./alexnet_test_gpu.sh`
- VGG test: `docker run --gpus all -it kaizen-gpu ./vgg_gpu.sh`
- LeNet test: `docker run --gpus all -it kaizen-gpu ./lenet_test_gpu.sh`

## Building Without Docker

If you prefer to build directly on your host system, you'll need:

1. CUDA Toolkit installed
2. CMake 3.10+
3. GMP development libraries

Then run:

```bash
mkdir -p build_gpu
cd build_gpu
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSESHA3=0 ..
make -j$(nproc)
```

## Troubleshooting

If you encounter issues:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Test simple CUDA application: `./test_gpu`
4. Verify Docker NVIDIA runtime: `docker info | grep nvidia`

## Performance Considerations

- The Docker-based approach has some overhead but is the easiest way to get started
- For best performance, build directly on your host system
- Adjust the batch size based on your GPU memory capacity 