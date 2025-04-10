# Kaizen API Documentation

This document provides detailed information on the API for the Kaizen zero-knowledge proof library for neural networks.

## Table of Contents

1. [Introduction](#introduction)
   - [Key Concepts](#key-concepts)
   - [Architecture Overview](#architecture-overview)
   - [Version Information](#version-information)

2. [Core Components](#core-components)
   - [Field Element Operations](#field-element-operations)
   - [Polynomial Operations](#polynomial-operations)
   - [Timing Utilities](#timing-utilities)
   - [Merkle Trees](#merkle-trees)
   - [Hashing Functions](#hashing-functions)
   
3. [Neural Network Components](#neural-network-components)
   - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
   - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
   - [Quantization](#quantization)
   
4. [Zero-Knowledge Proof Components](#zero-knowledge-proof-components)
   - [GKR Protocol](#gkr-protocol)
   - [Polynomial Commitment Schemes](#polynomial-commitment-schemes)
   - [Prover Functions](#prover-functions)
   - [Verifier Functions](#verifier-functions)
   
5. [Profiling Tools](#profiling-tools)
   - [Profiler Class](#profiler-class)
   - [Profiling Macros](#profiling-macros)
   - [Timer Utilities](#timer-utilities)

6. [Integration Guidelines](#integration-guidelines)
   - [Integrating with Existing ML Frameworks](#integrating-with-existing-ml-frameworks)
   - [Deploying in Production](#deploying-in-production)

7. [Performance Considerations](#performance-considerations)
   - [Optimizing Proof Generation](#optimizing-proof-generation)
   - [Memory Usage](#memory-usage)
   - [Parallelization](#parallelization)

8. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Debugging Tips](#debugging-tips)

9. [FAQ](#faq)

---

## Introduction

Kaizen is a comprehensive framework for implementing zero-knowledge proofs for neural network training and inference. It enables verifiable computation for machine learning models without revealing sensitive data.

### Key Concepts

- **Zero-Knowledge Proofs (ZKP)**: Cryptographic protocols that allow one party (the prover) to prove to another party (the verifier) that a statement is true without revealing any information beyond the validity of the statement itself.

- **General-purpose Succinct Non-interactive Arguments of Knowledge (GKR Protocol)**: A protocol for generating proofs of computation that are significantly smaller than the computation itself.

- **Polynomial Commitment Schemes**: Cryptographic schemes that allow a prover to commit to a polynomial and later reveal evaluations of the polynomial at specific points.

### Architecture Overview

The Kaizen library is structured as follows:

```
Kaizen
├── Core Mathematical Components
│   ├── Field Elements
│   ├── Polynomials
│   └── Hashing Functions
├── Neural Network Components
│   ├── MLP Implementation
│   ├── CNN Implementation
│   └── Quantization
├── Zero-Knowledge Proof Components
│   ├── GKR Protocol
│   ├── Polynomial Commitments
│   ├── Prover
│   └── Verifier
└── Profiling Tools
```

### Version Information

- **Current Version**: 1.0.0
- **Required Dependencies**:
  - C++11 or higher
  - GMP library (GNU Multiple Precision Arithmetic Library)
  - MCL library (for elliptic curves)

---

## Core Components

### Field Element Operations

The field element operations are defined in `fieldElement.hpp` and implemented in `fieldElement.cpp`. These provide the mathematical foundation for all cryptographic operations.

#### Class: `F` (Field Element)

Represents elements in a finite field used throughout the library. The field is defined over a prime modulus to ensure security properties.

**Constructor:**
```cpp
F();                 // Default constructor (zero element)
F(int64_t val);      // Constructor from integer
```

**Methods:**
```cpp
F operator+(const F& other) const;  // Addition in the field
F operator-(const F& other) const;  // Subtraction in the field
F operator*(const F& other) const;  // Multiplication in the field
F operator/(const F& other) const;  // Division in the field
F pow(uint64_t power) const;        // Exponentiation in the field
F inverse() const;                  // Multiplicative inverse in the field
bool operator==(const F& other) const; // Equality comparison
```

**Example Usage:**
```cpp
// Perform field arithmetic
F a(5);
F b(7);
F c = a * b;  // c = 35 mod p, where p is the field modulus
F d = c.inverse();  // d = c^(-1) mod p
bool equal = (c * d == F(1));  // Should be true
```

#### Utility Functions:

```cpp
F random_field_element();  // Generates a random field element
```

**Performance Considerations:**
- Field operations are the most frequent computations in the library
- For large proofs, optimizing field arithmetic can significantly improve performance
- Consider using vectorized implementations for bulk field operations

### Polynomial Operations

Defined in `polynomial.h` and implemented in `polynomial.cpp`. The polynomial operations are essential for the GKR protocol and polynomial commitment schemes.

#### Class: `Polynomial`

Represents a polynomial over the field, stored as a vector of coefficients.

**Constructor:**
```cpp
Polynomial(vector<F> coefficients);  // Construct from coefficients
Polynomial(int degree);              // Construct polynomial of specific degree
```

**Methods:**
```cpp
F evaluate(F point) const;            // Evaluate polynomial at a point
Polynomial operator+(const Polynomial& other) const;  // Addition of polynomials
Polynomial operator*(const Polynomial& other) const;  // Multiplication of polynomials
vector<F> getCoefficients() const;    // Get coefficients
Polynomial derivative() const;        // Get the derivative polynomial
int getDegree() const;                // Get degree of the polynomial
```

**Example Usage:**
```cpp
// Create a polynomial p(x) = 3x^2 + 2x + 1
vector<F> coeffs = {F(1), F(2), F(3)};
Polynomial p(coeffs);

// Evaluate at x = 5
F result = p.evaluate(F(5));  // result = 3*5^2 + 2*5 + 1 = 3*25 + 10 + 1 = 86

// Compute derivative: p'(x) = 6x + 2
Polynomial p_prime = p.derivative();
```

#### Functions:
```cpp
Polynomial interpolate(vector<F> x_points, vector<F> y_points);  // Lagrange interpolation
vector<F> evaluate_batch(const Polynomial& p, vector<F> points); // Batch evaluation
```

**Performance Considerations:**
- For high-degree polynomials, consider using FFT-based multiplication
- Batch evaluation can significantly reduce computational overhead for multiple evaluations
- When working with sparse polynomials, consider specialized representations

### Timing Utilities

Defined in `timer.hpp` and implemented in `timer.cpp`. These utilities provide precise timing measurements for performance analysis.

#### Class: `timer`

Provides high-resolution timing functionality for performance measurement.

**Methods:**
```cpp
void start();                // Start timer
void stop();                 // Stop timer
void clear();                // Reset timer
double elapse_sec() const;   // Get elapsed time in seconds
double elapse_ms() const;    // Get elapsed time in milliseconds
```

**Example Usage:**
```cpp
timer t;
t.start();
// Perform computation
expensive_operation();
t.stop();
printf("Operation took %.6f seconds\n", t.elapse_sec());
```

### Merkle Trees

Defined in `merkle_tree.h` and implemented in `merkle_tree.cpp`. Merkle trees are used for efficient commitment to large datasets.

#### Class: `MerkleTree`

Implements a Merkle tree for cryptographic commitments to a set of values (leaves).

**Constructor:**
```cpp
MerkleTree(const vector<F>& leaves);  // Construct from leaf values
```

**Methods:**
```cpp
vector<F> getRoot() const;           // Get Merkle root
vector<F> getProof(size_t index) const;  // Generate proof for leaf
bool verify(const vector<F>& root, size_t index, const F& leaf, const vector<F>& proof) const;  // Verify proof
```

**Example Usage:**
```cpp
// Create a Merkle tree from a list of values
vector<F> values = {F(1), F(2), F(3), F(4)};
MerkleTree tree(values);

// Get the root hash
vector<F> root = tree.getRoot();

// Generate a proof for the second value
vector<F> proof = tree.getProof(1);

// Verify the proof
bool valid = tree.verify(root, 1, values[1], proof);
```

**Visual Representation:**
```
        Root Hash
       /        \
     H01        H23
    /   \      /   \
   H0   H1    H2   H3
   |    |     |    |
  val0 val1  val2 val3
```

### Hashing Functions

Defined in `mimc.h` and implemented in `mimc.cpp`. These provide cryptographic hash functions optimized for zero-knowledge proofs.

#### Functions:
```cpp
F mimcHash(const F& left, const F& right);  // MiMC hash function (ZKP-friendly)
vector<F> sha256(const vector<F>& input);   // SHA-256 hash function
```

**Example Usage:**
```cpp
// Hash two field elements using MiMC
F a(123);
F b(456);
F hash_result = mimcHash(a, b);

// Hash a vector of field elements using SHA-256
vector<F> data = {F(1), F(2), F(3)};
vector<F> hash = sha256(data);
```

**Security Considerations:**
- MiMC is designed to be efficient in ZKP systems but may have different security properties than traditional hash functions
- For applications requiring standard cryptographic security, use the SHA-256 implementation

---

## Neural Network Components

### Multi-Layer Perceptron (MLP)

Defined in `MLP.h` and implemented in `MLP.cpp`. The MLP implementation provides the foundation for feedforward neural networks.

#### Structures:

**`feedforward_transcript`**
```cpp
struct feedforward_transcript {
    vector<vector<vector<F>>> Z;           // Pre-activation values per layer
    vector<vector<vector<F>>> Z_act;       // Post-activation values per layer
    vector<vector<vector<F>>> W;           // Weights per layer
    vector<vector<F>> b;                   // Biases per layer
    vector<vector<F>> X;                   // Input data
    // Additional fields for intermediate computation results
};
```

**`backpropagation_transcript`**
```cpp
struct backpropagation_transcript {
    vector<vector<vector<F>>> Z;           // Pre-activation values per layer
    vector<vector<vector<F>>> Z_act;       // Post-activation values per layer
    vector<vector<vector<F>>> W;           // Weights per layer
    vector<vector<F>> b;                   // Biases per layer
    vector<vector<F>> X;                   // Input data
    vector<vector<F>> y;                   // Target values
    vector<vector<vector<F>>> dW;          // Weight gradients
    // Additional fields for computation results during backpropagation
};
```

#### Functions:
```cpp
// Perform forward pass through the network
feedforward_transcript feed_forward_transcript();

// Perform backpropagation for training
backpropagation_transcript back_propagation_transcript();

// Core matrix operations
vector<vector<F>> _matrix_mul(vector<vector<F>> M1, vector<vector<F>> M2);  // Matrix multiplication
vector<vector<F>> _relu(vector<vector<F>> M1);  // ReLU activation function
vector<vector<F>> _softmax(vector<vector<F>> M1);  // Softmax activation function
```

**Example: Creating a Simple MLP**
```cpp
// Define a 2-layer MLP with input size 4, hidden size 10, output size 3
// Input data
vector<vector<F>> X = create_input_data(batch_size, 4);

// Initialize weights and biases
vector<vector<vector<F>>> W;
vector<vector<F>> b;

// First layer: 4 -> 10
W.push_back(initialize_weights(10, 4));
b.push_back(initialize_biases(10));

// Second layer: 10 -> 3
W.push_back(initialize_weights(3, 10));
b.push_back(initialize_biases(3));

// Perform forward pass
feedforward_transcript ff = feed_forward_transcript();

// Get output from the final layer
vector<vector<F>> output = ff.Z_act.back();
```

**Neural Network Architecture Visualization:**
```
Input Layer     Hidden Layer    Output Layer
   [x1]             [h1]            [o1]
   [x2]             [h2]            [o2]
   [x3]             [h3]            [o3]
   [x4]             ...
                   [h10]
```

### Convolutional Neural Network (CNN)

Defined in `CNN.h` and implemented in `CNN.cpp`. The CNN implementation provides convolutional layers, pooling, and other operations needed for image processing tasks.

#### Structures:

**`convolution_layer`**
```cpp
struct convolution_layer {
    int Batch_size;  // Batch size for processing
    int chout;       // Number of output channels
    int chin;        // Number of input channels
    int n;           // Input dimension (width/height for square inputs)
    int w;           // Weight dimension
    int window;      // Convolution window size
    int padded_w;    // Padded width after applying padding
    int idx;         // Layer index in the network
    vector<vector<F>> W;  // Weights
    vector<vector<F>> X;  // Input data
    // Additional fields for intermediate computations
};
```

**`relu_layer`**
```cpp
struct relu_layer {
    int Q_max;       // Quantization parameter for fixed-point arithmetic
    vector<F> input, new_input;  // Input values and processed input
    vector<F> output, temp_output;  // Output values and intermediate outputs
    vector<F> input_bits;  // Bit representation for input
    vector<F> most_significant_bits;  // MSB values for sign detection
};
```

**`avg_layer`**
```cpp
struct avg_layer {
    vector<vector<F>> U, Out, Sum, Remainder, Out_temp;  // Computation values
    int Batch_size, chout, w, n, window, padded_w;  // Dimensions
};
```

#### Functions:
```cpp
// Core CNN operations
vector<vector<vector<F>>> convolution(
    vector<vector<vector<F>>> input,       // Input tensor [batch][channel][pixel]
    vector<vector<vector<vector<F>>>> w    // Weight tensor [out_ch][in_ch][height][width]
);

vector<vector<vector<F>>> relu_activation(vector<vector<vector<F>>> input);  // ReLU activation

vector<vector<vector<F>>> avg_pooling(
    vector<vector<vector<F>>> input,       // Input tensor
    int window_size                        // Pooling window size
);

// Flatten 3D tensor to 2D for fully connected layers
vector<vector<F>> flatten(vector<vector<vector<F>>> input);
```

**Example: Creating a Simple CNN**
```cpp
// Define a simple CNN for MNIST-style image classification
// Input: 28x28x1 images
// Conv1: 5x5 kernels, 32 filters
// Pool1: 2x2 average pooling
// Conv2: 5x5 kernels, 64 filters
// Pool2: 2x2 average pooling
// FC: 7x7x64 -> 10 (flattened output to 10 classes)

// Input data [batch_size][1][28*28]
vector<vector<vector<F>>> input = load_mnist_batch(batch_size);

// First convolution layer
vector<vector<vector<vector<F>>>> conv1_weights = initialize_conv_weights(32, 1, 5, 5);
auto conv1_output = convolution(input, conv1_weights);
auto conv1_activated = relu_activation(conv1_output);

// First pooling layer
auto pool1_output = avg_pooling(conv1_activated, 2);

// Second convolution layer
vector<vector<vector<vector<F>>>> conv2_weights = initialize_conv_weights(64, 32, 5, 5);
auto conv2_output = convolution(pool1_output, conv2_weights);
auto conv2_activated = relu_activation(conv2_output);

// Second pooling layer
auto pool2_output = avg_pooling(conv2_activated, 2);

// Flatten for fully connected layer
auto flattened = flatten(pool2_output);

// Fully connected layer
vector<vector<F>> fc_weights = initialize_fc_weights(10, 7*7*64);
auto output = _matrix_mul(flattened, fc_weights);
```

**CNN Architecture Visualization:**
```
Input -> Conv1 -> ReLU -> Pool1 -> Conv2 -> ReLU -> Pool2 -> Flatten -> FC -> Output
28x28   24x24    24x24   12x12    8x8      8x8     4x4      1x1024    1x10
 x1      x32      x32     x32     x64      x64     x64
```

### Quantization

Defined in `quantization.h` and implemented in `quantization.cpp`. Quantization converts floating-point values to fixed-point representations, which are easier to handle in ZKP systems.

#### Functions:
```cpp
// Convert floating-point values to fixed-point
vector<F> quantize(
    const vector<F>& input,  // Floating-point input values
    int bits                 // Number of bits for quantization
);

// Convert fixed-point values back to floating-point
vector<F> dequantize(
    const vector<F>& quantized,  // Quantized fixed-point values
    int bits                     // Number of bits used in quantization
);
```

**Example:**
```cpp
// Original floating-point values
vector<F> float_values = {F(0.125), F(0.375), F(0.625), F(0.875)};

// Quantize to 8-bit fixed-point
vector<F> quantized = quantize(float_values, 8);

// Dequantize back to floating-point
vector<F> recovered = dequantize(quantized, 8);

// Check quantization error
for (size_t i = 0; i < float_values.size(); i++) {
    F error = float_values[i] - recovered[i];
    cout << "Original: " << float_values[i] << ", Recovered: " << recovered[i] 
         << ", Error: " << error << endl;
}
```

**Performance and Accuracy Trade-offs:**
- Higher bit precision reduces quantization error but increases proof size
- 8-bit quantization is often sufficient for inference tasks
- Consider using different bit precision for different layers based on sensitivity

---

## Zero-Knowledge Proof Components

### GKR Protocol

Defined in `GKR.h` and implemented in `GKR.cpp`. The GKR protocol is a powerful interactive proof system for verifying the execution of layered arithmetic circuits.

#### Structures:

**`proof`**
```cpp
struct proof {
    int type;                // Proof type identifier
    vector<F> evaluations;   // Polynomial evaluations at challenge points
    vector<F> commitments;   // Commitments to polynomials
    vector<F> randomness;    // Random challenges used in the protocol
    // Additional fields for different proof types
};
```

#### Functions:
```cpp
// Generate a sumcheck proof for a specific circuit layer
proof generate_sumcheck_proof(
    vector<vector<F>>& circuit,           // Circuit description
    vector<F>& input_assignment           // Input values
);

// Verify a sumcheck proof
bool verify_sumcheck_proof(
    const proof& p,                      // The proof to verify
    vector<vector<F>>& circuit           // Circuit description
);

// Generate a complete GKR proof for multi-layer circuits
proof generate_gkr_proof(
    vector<vector<vector<F>>>& circuit_layers,  // All circuit layers
    vector<F>& input                            // Input values
);

// Verify a complete GKR proof
bool verify_gkr_proof(
    const proof& p,                            // The proof to verify
    vector<vector<vector<F>>>& circuit_layers  // All circuit layers
);
```

**How GKR Works:**

1. The circuit is represented as layers, where each layer is a collection of gates.
2. The prover commits to the values of all gates in each layer.
3. The prover and verifier engage in a sumcheck protocol to reduce verification of the entire layer to verification of a single gate.
4. This process is repeated for each layer, ultimately reducing to a check of the input layer.

**Example Usage:**
```cpp
// Represent a neural network as a layered arithmetic circuit
vector<vector<vector<F>>> circuit_layers = convert_nn_to_circuit(model);

// Input data
vector<F> input = flatten_input(input_data);

// Generate proof
proof p = generate_gkr_proof(circuit_layers, input);

// Verify the proof
bool is_valid = verify_gkr_proof(p, circuit_layers);
```

### Polynomial Commitment Schemes

Defined in `poly_commit.h` and implemented in `poly_commit.cpp`. Polynomial commitment schemes allow a prover to commit to a polynomial and later prove evaluations at specific points.

#### Structures:

**`commitment`**
```cpp
struct commitment {
    vector<F> com;  // Commitment value
    Polynomial p;   // Committed polynomial
};
```

#### Functions:
```cpp
// Commit to a polynomial
commitment commit(
    const Polynomial& p  // The polynomial to commit to
);

// Open the commitment at a specific point
proof open(
    const commitment& c,  // The commitment
    const F& point        // The point at which to evaluate
);

// Verify an opening
bool verify_opening(
    const commitment& c,  // The commitment
    const F& point,       // The evaluation point
    const F& value,       // The claimed evaluation
    const proof& p        // The opening proof
);
```

**Supported Commitment Schemes:**

1. **Orion** (PC_TYPE = 1): Based on multilinear extensions and suitable for high-degree polynomials.
2. **Virgo** (PC_TYPE = 2): Based on inner product arguments, offering constant-size proofs at the cost of longer verification time.

**Example Usage:**
```cpp
// Create a polynomial
vector<F> coeffs = {F(1), F(3), F(5)};  // p(x) = 5x^2 + 3x + 1
Polynomial poly(coeffs);

// Commit to the polynomial
commitment com = commit(poly);

// Open the commitment at x = 7
F point(7);
proof opening = open(com, point);

// Verify the opening
F expected_value = poly.evaluate(point);  // Expected: 5*7^2 + 3*7 + 1 = 267
bool is_valid = verify_opening(com, point, expected_value, opening);
```

### Prover Functions

Defined in `prover.h` and implemented in `prover.cpp`. These functions tie together the various components to generate proofs for specific neural network architectures.

#### Functions:
```cpp
// Generate a proof for MLP inference
proof prove_MLP(
    const vector<vector<F>>& input,             // Input data
    const vector<vector<vector<F>>>& weights    // Network weights
);

// Generate a proof for CNN inference
proof prove_CNN(
    const vector<vector<vector<F>>>& input,             // Input data (images)
    const vector<vector<vector<vector<F>>>>& weights    // CNN weights
);
```

**Example Usage:**
```cpp
// Setup MLP model
vector<vector<F>> input = load_input_data();
vector<vector<vector<F>>> weights = load_weights();

// Generate proof
proof p = prove_MLP(input, weights);

// This proof can now be transmitted to a verifier
```

**Performance Considerations:**
- Proof generation is significantly more expensive than verification
- For large networks, consider parallelizing the proof generation
- Use optimized field arithmetic for better performance

### Verifier Functions

Defined in `verifier.h` and implemented in `verifier.cpp`. These functions verify the correctness of the proofs generated for neural network computations.

#### Functions:
```cpp
// Verify a proof for MLP inference
bool verify_MLP(
    const proof& p,                       // The proof to verify
    const vector<F>& output_commitment    // Commitment to expected output
);

// Verify a proof for CNN inference
bool verify_CNN(
    const proof& p,                       // The proof to verify
    const vector<F>& output_commitment    // Commitment to expected output
);
```

**Example Usage:**
```cpp
// Receive a proof from the prover
proof p = receive_proof();

// Receive or compute the expected output commitment
vector<F> output_commitment = receive_output_commitment();

// Verify the proof
bool is_valid = verify_MLP(p, output_commitment);

if (is_valid) {
    cout << "The proof is valid! The computation was performed correctly." << endl;
} else {
    cout << "The proof is invalid! The computation may be incorrect." << endl;
}
```

**Security Considerations:**
- The verifier should generate their own randomness for challenge points
- Verification should be performed in a secure environment
- For critical applications, consider using additional security measures

---

## Profiling Tools

### Profiler Class

Defined in `profiler.hpp` and implemented in `profiler.cpp`. The profiler provides detailed performance measurements to identify bottlenecks.

#### Class: `Profiler`

Provides comprehensive profiling functionality to measure performance of different code segments.

**Methods:**
```cpp
// Start timing a specific segment
void start(const std::string& segment_name);

// Stop timing a specific segment
void stop(const std::string& segment_name);

// Print detailed profiling report
void print_report() const;

// Reset all timers for a fresh profiling session
void reset();
```

**Example Output:**
```
===== PERFORMANCE PROFILING REPORT =====
Segment                         Total Time (s)       Calls       Avg Time (ms)
---------------------------------------------------------------------------
matrix_multiplication           12.345678            500         24.691
convolution                     8.765432             50          175.309
relu_activation                 0.987654             550         1.795
softmax                         0.543210             50          10.864
---------------------------------------------------------------------------
TOTAL                           22.641974

Bottleneck Analysis:
Primary bottleneck: matrix_multiplication (54.53% of total time)
```

### Profiling Macros

Macros for simplified profiling syntax:

```cpp
// Macro for easy profiling of code blocks using RAII
#define PROFILE_SCOPE(name)  // Automatically starts and stops timing
```

**Using the Macro:**
```cpp
void expensive_function() {
    PROFILE_SCOPE("expensive_function");
    
    // Nested profiling is supported
    {
        PROFILE_SCOPE("first_part");
        // First part of the function
    }
    
    {
        PROFILE_SCOPE("second_part");
        // Second part of the function
    }
}
```

### Timer Utilities

Provides low-level timing functionality used by the profiler.

```cpp
// Helper class for automatic profiling using RAII
class ProfilerScope {
public:
    ProfilerScope(const std::string& name);  // Starts timing
    ~ProfilerScope();  // Stops timing when object goes out of scope
    
private:
    std::string name_;  // Name of the profiled segment
};
```

#### Global Instance:
```cpp
// Global profiler instance that can be accessed from anywhere
extern Profiler g_profiler;
```

**Example: Detailed Profiling of an Algorithm:**
```cpp
#include "profiler.hpp"

void train_network() {
    PROFILE_SCOPE("train_network");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        PROFILE_SCOPE("epoch");
        
        {
            PROFILE_SCOPE("forward_pass");
            // Forward pass code
        }
        
        {
            PROFILE_SCOPE("backward_pass");
            // Backward pass code
        }
        
        {
            PROFILE_SCOPE("weight_update");
            // Weight update code
        }
    }
    
    g_profiler.print_report();
}
```

---

## Integration Guidelines

### Integrating with Existing ML Frameworks

Kaizen can be integrated with existing machine learning frameworks by:

1. **Exporting Trained Models**: Export weights from frameworks like TensorFlow or PyTorch
2. **Converting Model Format**: Convert to Kaizen's internal representation
3. **Generating Proofs**: Use Kaizen's API to generate proofs for the converted model

**Example: Integrating with PyTorch**
```cpp
// Python side (PyTorch):
// model = ... # PyTorch model
// torch.save(model.state_dict(), 'model_weights.pt')

// C++ side (Kaizen):
// Load weights from the exported file
vector<vector<vector<F>>> weights = load_weights_from_pytorch_format("model_weights.pt");

// Generate proof for inference
vector<vector<F>> input = load_input_data();
proof p = prove_MLP(input, weights);
```

### Deploying in Production

When deploying Kaizen in production environments, consider:

1. **Security**: Ensure that the prover's private data remains secure
2. **Performance**: Optimize for the specific use case (proof generation or verification)
3. **Integration**: Use appropriate APIs for your system architecture
4. **Monitoring**: Implement logging and monitoring for proof generation and verification

**Recommended Deployment Patterns:**

1. **Client-Server Model**: Client generates proofs, server verifies them
2. **Batch Processing**: Generate proofs offline for later verification
3. **Hybrid Systems**: Combine trusted execution environments with ZKP for enhanced security

---

## Performance Considerations

### Optimizing Proof Generation

Proof generation is computationally intensive. Consider these optimizations:

1. **Field Arithmetic**: Use optimized libraries for field operations
2. **Parallelization**: Parallelize proof generation across multiple cores
3. **Polynomial Operations**: Use FFT for large polynomial multiplications
4. **Memory Management**: Minimize allocations in critical sections

**Performance Benchmarks:**
| Model Size | Proof Generation Time | Verification Time | Proof Size |
|------------|----------------------|-------------------|------------|
| Small MLP (2 layers, 64 neurons) | 0.5-1s | <0.1s | ~10KB |
| Medium CNN (LeNet) | 5-10s | ~0.5s | ~50KB |
| Large CNN (AlexNet) | 60-120s | ~2s | ~200KB |

### Memory Usage

Memory optimizations for proof generation:

1. **Streaming Computation**: Process data in chunks where possible
2. **In-place Operations**: Modify data in-place when possible
3. **Memory Pooling**: Reuse memory allocations for similar-sized objects
4. **Compact Representations**: Use compact data structures for proof elements

### Parallelization

The library supports parallelization through:

1. **Multi-threading**: Use multiple threads for proof generation
2. **Batch Processing**: Process multiple inputs in parallel
3. **SIMD Instructions**: Leverage vectorized operations for field arithmetic

**Example: Enabling Multi-threading**
```cpp
// Set the number of threads for proof generation
int threads = 4;
set_num_threads(threads);

// Now proof generation will use 4 threads where possible
proof p = prove_MLP(input, weights);
```

---

## Troubleshooting

### Common Issues

**Verification Failures**:
- **Issue**: Proof verification fails unexpectedly
- **Possible Causes**:
  - Incorrect input data format
  - Mismatch between prover and verifier parameters
  - Numerical precision errors in floating-point to field element conversion
- **Solution**: Verify input data consistency and parameter matching

**Performance Problems**:
- **Issue**: Proof generation is too slow
- **Possible Causes**:
  - Unoptimized field arithmetic
  - Excessive memory allocations
  - Large model size
- **Solution**: Use profiling tools to identify bottlenecks and apply targeted optimizations

**Memory Consumption**:
- **Issue**: High memory usage during proof generation
- **Possible Causes**:
  - Storing all intermediate values
  - Inefficient polynomial representations
- **Solution**: Implement streaming proof generation and optimize memory usage

### Debugging Tips

1. **Enable Verbose Logging**:
   ```cpp
   set_log_level(LogLevel::DEBUG);
   ```

2. **Check Intermediate Values**:
   ```cpp
   // Verify intermediate polynomial evaluations
   F expected = p.evaluate(challenge);
   F actual = evaluate_circuit_at_point(circuit, challenge);
   assert(expected == actual);
   ```

3. **Validate Input Data**:
   ```cpp
   // Ensure input data is correctly formatted
   validate_input(input);
   ```

4. **Compare Against Reference Implementation**:
   ```cpp
   // Compare against a non-ZKP reference
   vector<F> zkp_result = run_with_zkp(input);
   vector<F> ref_result = run_reference(input);
   compare_results(zkp_result, ref_result);
   ```

---

## FAQ

**Q: What is the maximum model size supported?**  
A: The library has been tested with models up to the size of AlexNet (~60M parameters). Larger models may require more memory and computation time.

**Q: Can I use custom neural network architectures?**  
A: Yes, as long as the operations can be expressed as arithmetic circuits. The library supports custom layer types through the circuit API.

**Q: How does the proof size scale with network size?**  
A: Proof size scales logarithmically with the computation size when using the GKR protocol with polynomial commitments.

**Q: Is the library compatible with quantized models?**  
A: Yes, the library includes quantization support specifically designed for ZKP systems.

**Q: Can I generate proofs for training, not just inference?**  
A: Yes, though proof generation for training is more computationally intensive due to the complexity of backpropagation.

**Q: What security guarantees does the library provide?**  
A: The library provides computational soundness based on the security of the underlying cryptographic primitives. The exact security level depends on the field size and commitment scheme used.

**Q: How can I optimize proof generation for my specific model?**  
A: Use the provided profiling tools to identify bottlenecks, then focus on optimizing those specific operations. Often, matrix multiplications and convolutions are the primary targets for optimization.

---

For more detailed examples, see the example files in the source code:
- `profiler_example.cpp`
- `profiling_example_mlp.cpp`
- `profiling_example_cnn.cpp` 

For questions not covered in this documentation, please open an issue on the project's repository. 