# Zero-Knowledge Proof of Training in Kaizen

This document explains how Kaizen implements a zero-knowledge proof of training (zkPoT) system for machine learning models.

## Overview

Kaizen provides a framework to prove that a machine learning model has been correctly trained on a dataset without revealing the training data or the internal model parameters. The system enables:

1. **Proof of Correct Training**: A prover can demonstrate that a model was trained using a specific algorithm and hyperparameters
2. **Verification of Model Properties**: A verifier can confirm properties of the model without seeing the training data
3. **Privacy Preservation**: Training data remains private, addressing data privacy concerns

## System Architecture

Kaizen implements zkPoT through several interconnected components:

### 1. Representation of ML Models

The system represents neural networks (MLP, CNN, etc.) as arithmetic circuits, where:
- Matrix operations are broken down into arithmetic operations over finite fields
- Activation functions (ReLU, softmax) are implemented using piecewise polynomial approximations
- The entire computation graph is expressed as a layered arithmetic circuit

### 2. Proof Generation Pipeline

#### Step 1: Training Representation
- The training algorithm (backpropagation) is represented as a sequence of arithmetic operations
- Operations are tracked in `MLP.cpp` and `CNN.cpp`, capturing forward and backward passes

#### Step 2: Circuit Construction
- The circuit representation is built in `circuit.cpp` and `GKR.cpp`
- Each training iteration is represented as a directed acyclic graph (DAG)
- Gates in the circuit represent operations like multiplication, addition, ReLU, etc.

#### Step 3: Commitment to Training Data
- Training data is committed using Merkle trees in `merkle_tree.cpp`
- This allows proving statements about the data without revealing it

#### Step 4: Sumcheck Protocol
- The GKR protocol uses the sumcheck protocol to efficiently verify circuit evaluation
- Implemented in `GKR.cpp` with supporting polynomial operations in `polynomial.cpp`

#### Step 5: Polynomial Commitments
- Model weights and activations are represented as polynomials
- Committed using schemes in `poly_commit.cpp` (supports both Virgo and Orion schemes)

#### Step 6: Zero-Knowledge Layer
- Additional randomization is added to ensure zero-knowledge property
- This prevents leakage of training data or model parameters

### 3. Verification Process

The verification process (in `verifier.cpp` and `pol_verifier.cpp`) allows a verifier to confirm:

1. **Consistency**: The model follows the claimed structure
2. **Correctness**: Each training step was performed correctly according to the claimed algorithm
3. **Adherence**: The training followed the specified hyperparameters and optimization strategy

## Neural Network to Circuit Transformation

Kaizen translates neural network operations into arithmetic circuits suitable for the GKR protocol. This transformation is a crucial step in enabling zero-knowledge proofs of training.

### Matrix Operations

1. **Matrix Multiplication**:
   - The fundamental operation in neural networks, $Z = W \times X + b$, is represented as a circuit
   - In `MLP.cpp`, matrix multiplication is proved using the function `prove_matrix2matrix`:
     ```cpp
     struct proof temp;
     if(i == 0){
         temp = prove_matrix2matrix(tr.W[0], tr.X);
     }
     else{
         temp = prove_matrix2matrix(tr.W[i], transpose(tr.Z_act[i-1]));
     }
     ```

2. **Convolution Operations**:
   - Convolutions in CNNs are handled by converting them to FFT-based multiplications:
     ```cpp
     Pr = prove_ifft_matrix(conv.Prod, reordered_r, previous_sum);
     Pr = prove_dot_x_prod(gkr_input, r, conv.fft_X[0].size(), conv.Batch_size, conv.chin, conv.chout);
     ```

### Activation Functions

1. **ReLU Transformation**:
   - The ReLU function ($\max(0, x)$) is represented using bit operations
   - The implementation extracts the sign bit to determine if a value is positive:
     ```cpp
     for(int i = 0; i < v.size(); i++){
         last_bits[0][i] = bits[i*256+254]; // Most significant bit indicates sign
         sign_poly.push_back(last_bits[0][i]);
     }
     ```
   - The arithmetic circuit multiplies the input by 0 or 1 based on the sign bit:
     ```cpp
     // Z_act = Z * (1 - sign_bit)
     struct proof temp = generate_GKR_proof(circuit_name, name, false);
     ```

2. **Softmax Implementation**:
   - Softmax is broken down into multiple operations:
     - Exponentiation computation using polynomial approximation
     - Division of each value by the sum of exponentiated values
     - The comment in the code explains: "Prove softmax, as softmax takes as input the Z vector of the last layer. Firstly shifts that vector with divisor 2^{Q*layers} so that all Z' values will be in small domain"

### Backpropagation Representation

1. **Gradient Computation**:
   - Backpropagation is also represented as arithmetic circuits:
     ```cpp
     void prove_relu_backprop(struct relu_layer_backprop relu_data, vector<F> &r, F &previous_sum){
         // Compute dx = dy * (1 - sign_bit)
         vector<F> nbits;
         for(int i = 0; i < relu_data.dx_prev.size(); i++){
             F num = F(1)-relu_data.most_significant_bits[i];
             nbits.push_back(num);
         }
         // ...
     }
     ```

2. **Weight Updates**:
   - Weight update operations are converted to circuit operations
   - The circuit verifies that $W_{t+1} = W_t - \alpha \cdot \nabla W$

### Quantization and Fixed-Point Arithmetic

To make the circuit representation efficient:

1. **Fixed-Point Representation**:
   - Real values are converted to fixed-point representations
   - This is seen in operations like:
     ```cpp
     divisor[i][j] = quantize(M_exp);
     quotient[i][j] = divide(F(1<<Q)*data[i][j], divisor[i][j]);
     ```

2. **Bit Decomposition**:
   - Many operations require bit-level manipulation, proven through bit decomposition circuits:
     ```cpp
     vector<F> bits = prepare_bit_vector(range_proof_data, 32);
     Transcript.push_back(_prove_bit_decomposition(bits, r, eval, 32));
     ```

### Circuit Construction

The complete neural network is represented as a sequence of interconnected subcircuits:

1. **Feedforward Pass**:
   - Represented by `prove_forward_propagation()` which chains multiple proofs together:
     ```cpp
     struct feedforward_proof prove_forward_propagation(){
         struct feedforward_transcript tr = feed_forward_transcript();
         struct feedforward_proof P;
         for(int i = 0; i < tr.W.size(); i++){
             // Prove matrix multiplication
             // Prove activation function
             // ...
         }
     }
     ```

2. **Custom Circuit Generation**:
   - Specific circuits are generated for different layer types and operations
   - Circuit files (`.pws` files) are referenced in the code:
     ```cpp
     string circuit_filename = "dot_product_circuit_" + to_string(conv.fft_X[0].size()) + "_" + to_string(conv.Batch_size) + "_" + to_string(conv.chin) + "_" + to_string(conv.chout) + ".pws";
     ```

This circuit representation allows Kaizen to generate zero-knowledge proofs for complex neural network training processes, enabling verification without revealing private data or model parameters.

## Mathematical Foundations

### Finite Field Arithmetic

The system operates over a prime field $\mathbb{F}_p$ where $p = 2^{61}-1$, implemented in `fieldElement.cpp`. Key mathematical properties include:

1. **Field Definition**:
   - Operations are performed modulo the prime $p = 2^{61}-1 = 2,305,843,009,213,693,951$
   - Elements are represented as pairs of 64-bit integers (real, img) for complex field elements

2. **Field Operations**:
   - Addition: $(a, b) + (c, d) = ((a + c) \bmod p, (b + d) \bmod p)$
   - Subtraction: $(a, b) - (c, d) = ((a - c) \bmod p, (b - d) \bmod p)$
   - Multiplication (complex): $(a, b) \times (c, d) = ((ac - bd) \bmod p, (ad + bc) \bmod p)$
   - The implementation includes optimizations for modular arithmetic:
     ```cpp
     unsigned long long fieldElement::myMod(unsigned long long int x) {
         return (x >> 61) + (x & mod);
     }
     ```

3. **Optimization Techniques**:
   - Platform-specific implementations using SIMD instructions for x86 platforms
   - Efficient Montgomery multiplication for modular arithmetic without expensive division
   - Batched operations using the `fieldElementPacked` class to process 4 field elements simultaneously

### The GKR Protocol

The GKR (Goldwasser-Kalai-Rothblum) protocol provides an interactive proof system for verifying computations. In Kaizen, it's used to verify the correct execution of the ML training circuit.

1. **Mathematical Framework**:
   - The computation is expressed as a layered arithmetic circuit $C$
   - Each layer $i$ has gates computing functions from the set $\{+, \times, -\}$
   - For each gate $g$ in layer $i$, values are computed based on outputs from layer $i-1$

2. **Multilinear Extensions**:
   - The protocol represents boolean functions as multilinear polynomials
   - Let $V_i: \{0,1\}^{k_i} \rightarrow \mathbb{F}$ be the function where $V_i(g)$ is the value of gate $g$ in layer $i$
   - The multilinear extension $\tilde{V}_i: \mathbb{F}^{k_i} \rightarrow \mathbb{F}$ extends this to the entire field

3. **Sumcheck Protocol**:
   - The prover claims that $\sum_{x \in \{0,1\}^n} f(x) = C$ for some polynomial $f$
   - Through interaction, the verifier is convinced of this claim without evaluating the sum
   - This requires only $O(n)$ communication and $O(n)$ verifier time
   - The sumcheck protocol is implemented in `GKR.cpp` through functions like:
     ```cpp
     void sumcheckInitPhase1(const F &assert_random);
     quadratic_poly sumcheckUpdatePhase1(const F &previousRandom);
     void sumcheckFinalize1(const F &previousRandom, F &claim);
     ```

4. **Protocol Execution**:
   - For each layer $i$ from output to input:
     - Prover and verifier engage in a sumcheck protocol for the polynomial relating layers $i$ and $i-1$
     - This reduces to evaluating $\tilde{V}_{i-1}$ at a random point
     - The process continues recursively until the input layer
   - The verification function in `pol_verifier.cpp` implements this:
     ```cpp
     void verify_gkr(struct proof P) {
         // Initialize with the output layer claim
         F temp_sum = F(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));
         // Process each layer
         for(int i = 0; i < layers; i++) {
             // Verify sumcheck protocol steps
             // ...
         }
     }
     ```

5. **Zero-Knowledge Property**:
   - The protocol is modified to ensure zero-knowledge by:
     - Adding random "masking" polynomials to hide actual values
     - Using commitments instead of direct values
     - Employing techniques to ensure the verifier learns nothing beyond the correctness of the computation

### Polynomial Commitment Schemes

To make the protocol non-interactive and zero-knowledge, Kaizen uses polynomial commitment schemes:

1. **Mathematical Concept**:
   - A commitment to a polynomial $p(X)$ allows the prover to later reveal evaluations $p(r)$ at specific points
   - The commitment is binding (prover can't change $p$ after committing) and hiding (verifier learns nothing about $p$ except the revealed evaluations)

2. **Orion Scheme** ($PC\_scheme = 1$):
   - Uses Reed-Solomon codes for polynomial encoding
   - The polynomial is encoded using Fast Fourier Transform (FFT):
     ```cpp
     vector<F> temp(2*arr.size());
     encode(arr.data(), temp.data(), arr.size());
     ```
   - Commitments are Merkle tree roots of the encoded polynomial
   - Query operations generate Merkle paths as proofs

3. **Virgo Scheme** ($PC\_scheme = 2$):
   - Uses inner product arguments over vector polynomials
   - More efficient for high-degree polynomials
   - Supported through the `commit_public` and `commit_private` functions

### Merkle Trees for Data Authentication

Merkle trees provide efficient commitments to large datasets:

1. **Mathematical Structure**:
   - A binary tree where each leaf contains a hash of a data block
   - Each internal node contains a hash of its two children
   - The root hash serves as a commitment to the entire dataset

2. **Implementation**:
   - Uses either SHA-256 or MiMC hash functions (selectable via `Commitment_hash`)
   - Tree construction:
     ```cpp
     merkle_tree::merkle_tree_prover::create_tree(buff[0].size(), buff, sizeof(__hhash_digest), true);
     ```
   - Merkle path generation for proofs:
     ```cpp
     void query(int col, int row, vector<vector<F>> &matrix, commitment mt, aggregation_witness &_data);
     ```

### Circuit-Specific Optimizations

Several mathematical optimizations make zkPoT practical:

1. **Matrix Multiplication Verification**:
   - Uses the Freivalds algorithm to probabilistically verify matrix multiplications
   - Reduces verification complexity from $O(n^3)$ to $O(n^2)$

2. **Lookup Proofs**:
   - Enables efficient verification of memory access operations
   - Uses a specialized protocol for proving correct array lookups

3. **Batching**:
   - Multiple training iterations are batched for efficient proving
   - Uses a technique similar to batch verification in signature schemes

## Technical Implementation Details

### Circuit Representation
- Gates are represented as structures in a directed acyclic graph (DAG)
- Each gate has a type (Mul, Add, Sub, etc.) and input connections
- The `DAG_gate` structure in `inputCircuit.hpp` defines:
  ```cpp
  struct DAG_gate {
      pair<int, u64> input0, input1;
      bool is_assert;
      gateType ty;
  };
  ```

### GKR Protocol Execution
- The protocol proceeds in phases:
  1. **Initialization**: Setting up the circuit and initial claims
  2. **Sumcheck Protocol**: Interactive proof to reduce multivariate claim to univariate
  3. **Polynomial Evaluation**: Verifying evaluations at random points
  4. **Final Verification**: Checking consistency of all proofs

- Key methods in the prover:
  ```cpp
  void sumcheckInit();
  quadratic_poly sumcheckUpdatePhase1(const F &previousRandom);
  quadratic_poly sumcheckUpdatePhase2(const F &previousRandom);
  void sumcheckFinalize1(const F &previousRandom, F &claim);
  void sumcheckFinalize2(const F &previousRandom, vector<F>::iterator claims);
  ```

### Polynomial Commitment Implementation
- For the Orion scheme (PC_scheme = 1):
  ```cpp
  // Encode polynomial using linear encoding
  vector<F> temp(2*arr.size());
  encode(arr.data(), temp.data(), arr.size());
  ```

- For tree-based commitments:
  ```cpp
  // Create Merkle tree for SHA-based commitment
  merkle_tree::merkle_tree_prover::create_tree(buff[0].size(), buff, sizeof(__hhash_digest), true);

  // Create Merkle tree for MiMC-based commitment
  merkle_tree::merkle_tree_prover::create_tree_mimc(_buff[0].size(), _buff, level, sizeof(__hhash_digest), true);
  ```

- Query operation to prove polynomial evaluations:
  ```cpp
  void query(int col, int row, vector<vector<F>> &matrix, commitment mt, aggregation_witness &_data) {
      // Generate Merkle path as proof
      // ...
  }
  ```

## Optimizations

Kaizen implements several optimizations to make zkPoT practical:

1. **Batching**: Multiple training iterations are batched for efficient proving
2. **Hardware Acceleration**: SIMD support for cryptographic operations
3. **Quantization**: Reduces precision requirements while maintaining accuracy
4. **Selective Verification**: Focus verification on critical parts of the training

## Limitations and Future Work

1. **Proof Size**: Current proofs can be large for complex models
2. **Proving Time**: Generating proofs is computationally intensive
3. **Activation Functions**: Some complex activations require approximation
4. **CUDA Support**: Current CUDA implementation has limitations:
   - Only supports a subset of operations on GPU
   - Memory transfers between CPU and GPU create overhead
   - Multi-GPU training not yet implemented
   - Complex circuit operations still executed on CPU

## Conclusion

Kaizen demonstrates that zero-knowledge proofs of training are feasible for practical machine learning models. The system enables new applications where training integrity must be verified while preserving data privacy.
