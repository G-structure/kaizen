# Kaizen: Zero-Knowledge Proofs for Neural Networks (zkPoT)

Kaizen is a framework for implementing and testing zero-knowledge proofs for neural network training and inference. It supports various neural network architectures including MLP, CNN, LeNet, AlexNet, and VGG.

## Overview

This project implements zero-knowledge proofs for neural networks, allowing for verifiable machine learning computations without revealing sensitive data. The framework includes:

- Multiple neural network architectures (MLP, CNN, LeNet, AlexNet, VGG)
- Profiling tools for performance optimization
- Support for different polynomial commitment schemes (Orion, Virgo)
- Integration with Merkle trees and SHA hash functions

## Installation

### Prerequisites

- CMake (3.10+)
- C++ compiler with C++11 support
- GMP library (GNU Multiple Precision Arithmetic Library)
- MCL library (for elliptic curves)

### Building

1. First, decompress the third-party dependencies:
   ```
   unzip 3rd.zip
   ```

2. Make the build script executable:
   ```
   chmod +x build.sh
   ```

3. Run the build script:
   ```
   ./build.sh
   ```

This will create a `build` directory, run CMake, and compile the project.

## Usage

### Running Tests with Pre-configured Scripts

You can run tests for specific neural network architectures using the provided scripts:

```
./{Architecture}_test.sh
```

Where `{Architecture}` can be:
- `lenet` - LeNet CNN architecture
- `alexnet` - AlexNet CNN architecture
- `vgg` - VGG CNN architecture

Each script runs the main program with different parameters to test various configurations.

### Running the Main Program Manually

You can also run the main program directly with specific parameters:

```
./build/src/main {MODEL} {BATCHES} {INPUT_FILTERS} {SHA_LEVELS} {PC_TYPE}
```

Parameters:
- `{MODEL}`: Neural network architecture (LENET, mAlexNet, VGG)
- `{BATCHES}`: Number of batches to process
- `{INPUT_FILTERS}`: Number of input filters/channels
- `{SHA_LEVELS}`: Number of levels of SHA hashes (-1 for no SHA hashes)
- `{PC_TYPE}`: Polynomial commitment type (1 for Orion, 2 for Virgo)

### Profiling Tools

The framework includes profiling tools to identify performance bottlenecks:

```
./build/src/run_profiler_test    # General profiler test
./build/src/run_mlp_profiler     # MLP-specific profiler
./build/src/run_cnn_profiler     # CNN-specific profiler
```

For more information on profiling, see `docs/PROFILER.md`.

## Project Structure

- `src/`: Source code
  - Neural network implementations (MLP.cpp, CNN.cpp)
  - Profiling tools (profiler.hpp, profiler.cpp)
  - Polynomial commitment schemes (poly_commit.cpp)
  - Verification tools (verifier.cpp)
  - Utility functions (utils.cpp, utils.hpp)
- `build/`: Build directory (created during compilation)
- Scripts for testing different architectures (lenet_test.sh, alexnet_test.sh, etc.)
