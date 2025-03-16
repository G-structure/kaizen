
# Profiler Results
```bash
root@localhost:~/projects/kaizen/build# cd /root/projects/kaizen && mkdir -p build && cd build && cmake .. && make run_profiler_test && ./src/run_profiler_test
-- Configuring done
-- Generating done
-- Build files have been written to: /root/projects/kaizen/build
Consolidate compiler generated dependencies of target run_profiler_test
[100%] Built target run_profiler_test
Running profiler test...
Epoch 1/2
Epoch 2/2

===== PERFORMANCE PROFILING REPORT =====
Segment                        Total Time (s)          Calls  Avg Time (ms)
---------------------------------------------------------------------------
total_execution                     28.628801              1      28628.801
epoch                               28.628794              2      14314.397
batch_processing                    28.628738              6       4771.456
mlp_training                        27.494914              6       4582.486
forward_pass                        27.481473              6       4580.246
matrix_multiplication               27.442579             24       1143.441
multiplication_loop                 27.390597             24       1141.275
layer1_forward                      27.342806              6       4557.134
cnn_training                         1.133807              6        188.968
cnn_backward                         0.900589              6        150.098
cnn_forward                          0.233202              6         38.867
pooling_layer1                       0.180485              6         30.081
layer2_forward                       0.138525              6         23.087
fully_connected                      0.051903              6          8.650
backward_pass                        0.013428              6          2.238
gradient_computation                 0.013397             18          0.744
input_layer_backward                 0.012987              6          2.164
convolution                          0.000764             12          0.064
conv_layer1                          0.000677              6          0.113
convolution_loop                     0.000650             12          0.054
hidden_layer_backward                0.000384              6          0.064
output_layer_forward                 0.000106              6          0.018
conv_layer2                          0.000105              6          0.018
output_layer_backward                0.000043              6          0.007
weight_update                        0.000028             18          0.002
---------------------------------------------------------------------------
TOTAL                              225.719784

Bottleneck Analysis:
Primary bottleneck: total_execution (12.68% of total time)
```

# CNN Profiler
```
root@localhost:~/projects/kaizen/build# cd /root/projects/kaizen && mkdir -p build && cd build && cmake .. && make run_cnn_profiler && ./src/run_cnn_profiler
-- Configuring done
-- Generating done
-- Build files have been written to: /root/projects/kaizen/build
Consolidate compiler generated dependencies of target run_cnn_profiler
[100%] Built target run_cnn_profiler
Running CNN profiling test...
CNN forward pass completed.
Final output sample: 0

===== PERFORMANCE PROFILING REPORT =====
Segment                        Total Time (s)          Calls  Avg Time (ms)
---------------------------------------------------------------------------
total_cnn_execution                  0.033518              1         33.518
convolution                          0.031728              2         15.864
conv_computation                     0.031604              2         15.802
simple_convolution                   0.030413            152          0.200
conv_loop                            0.029766            152          0.196
conv_prod                            0.021292          26624          0.001
conv_layer1                          0.017716              1         17.716
conv_layer2                          0.014273              1         14.273
avg_pooling                          0.001290              2          0.645
pool_computation                     0.001246              2          0.623
pool_layer1                          0.000949              1          0.949
add_matrix                           0.000948            152          0.006
compute_avg                          0.000734           1552          0.000
output_allocation                    0.000448            152          0.003
pool_layer2                          0.000345              1          0.345
relu_3d                              0.000243              2          0.121
data_initialization                  0.000191              1          0.191
conv_memory_allocation               0.000118              2          0.059
pool_allocation                      0.000038              2          0.019
---------------------------------------------------------------------------
TOTAL                                0.216859

Bottleneck Analysis:
Primary bottleneck: total_cnn_execution (15.46% of total time)
```

# MLP Profiler
```
root@localhost:~/projects/kaizen/build# cd /root/projects/kaizen && mkdir -p build && cd build && cmake .. && make run_mlp_profiler && ./src/run_mlp_profiler
-- Configuring done
-- Generating done
-- Build files have been written to: /root/projects/kaizen/build
Consolidate compiler generated dependencies of target run_mlp_profiler
[ 25%] Building CXX object src/CMakeFiles/run_mlp_profiler.dir/run_mlp_profiler.cpp.o
[ 50%] Building CXX object src/CMakeFiles/run_mlp_profiler.dir/profiler.cpp.o
[ 75%] Building CXX object src/CMakeFiles/run_mlp_profiler.dir/timer.cpp.o
[100%] Linking CXX executable run_mlp_profiler
[100%] Built target run_mlp_profiler
Running MLP profiling test...
Sample output: 211.406

===== PERFORMANCE PROFILING REPORT =====
Segment                        Total Time (s)          Calls  Avg Time (ms)
---------------------------------------------------------------------------
total_mlp_execution                  0.008910              1          8.910
complete_mlp_forward                 0.008708              1          8.708
feedforward                          0.008706              1          8.706
matrix_multiplication                0.008415              2          4.207
matrix_computation                   0.008362              2          4.181
layer_1                              0.005613              1          5.613
layer_2                              0.003028              1          3.028
data_initialization                  0.000170              1          0.170
add_bias                             0.000141              2          0.070
relu_activation                      0.000071              1          0.071
matrix_memory_allocation             0.000046              2          0.023
---------------------------------------------------------------------------
TOTAL                                0.052170

Bottleneck Analysis:
Primary bottleneck: total_mlp_execution (17.08% of total time)
```

# Profiling Guide for Neural Network Training

This guide explains how to use the built-in profiler to identify performance bottlenecks in the neural network training code.

## Overview

Based on code examination, potential bottlenecks in the training pipeline include:

1. **Matrix Multiplications**: The most computationally intensive operations in both MLP and CNN.
2. **Convolution Operations**: Especially in CNN, these are typically slow.
3. **Gradient Calculations**: In backpropagation, calculating gradients can be expensive.
4. **Memory Operations**: Allocating and copying large tensors.
5. **Activation Functions**: While usually not the main bottleneck, these can add up.

## How to Use the Profiler

The profiler helps measure time spent in different parts of your code.

### Basic Usage

1. Include the profiler header:
```cpp
#include "profiler.hpp"
```

2. Instrument code with timing blocks:
```cpp
// Method 1: Using start/stop
g_profiler.start("operation_name");
// Your expensive operation here
g_profiler.stop("operation_name");

// Method 2: Using automatic scoping (preferred)
{
    PROFILE_SCOPE("operation_name");
    // Your expensive operation here
}
```

3. Print the profiling report:
```cpp
g_profiler.print_report();
```

### Profiling Specific Components

#### 1. Matrix Operations

Add profiling to `matrix_mul` and related functions in MLP.cpp:

```cpp
vector<vector<F>> _matrix_mul(vector<vector<F>> M1, vector<vector<F>> M2){
    PROFILE_SCOPE("matrix_multiplication");
    // Existing code...
}
```

#### 2. Convolution Operations

Add profiling to convolution functions in CNN.cpp:

```cpp
vector<vector<vector<F>>> convolution(vector<vector<vector<F>>> input, vector<vector<vector<vector<F>>>> w){
    PROFILE_SCOPE("convolution");
    // Existing code...
}
```

#### 3. Forward Pass

Add profiling to the feed-forward functions:

```cpp
struct feedforward_transcript feed_forward_transcript(){
    PROFILE_SCOPE("feed_forward_pass");
    // Existing code...
}
```

#### 4. Backward Pass

Add profiling to backpropagation:

```cpp
struct backpropagation_transcript back_propagation_transcript(){
    PROFILE_SCOPE("back_propagation_pass");
    // Existing code...
}
```

#### 5. Layer-by-Layer Analysis

For more detailed analysis, profile individual layer operations:

```cpp
// In CNN.cpp or MLP.cpp where layers are processed
{
    PROFILE_SCOPE("layer1_forward");
    // Layer 1 code
}
{
    PROFILE_SCOPE("layer2_forward");
    // Layer 2 code
}
```

### Recommended Profiling Strategy

1. **Start with high-level profiling**: Profile major components first (forward pass, backward pass)
2. **Drill down into bottlenecks**: Once you identify a slow component, add more detailed profiling
3. **Compare algorithms**: If you have alternative implementations, profile both to compare
4. **Memory vs. Speed**: Remember that sometimes memory operations can be the bottleneck

## Profiling in Main Function

Add this to your main.cpp to enable profiling for a complete training run:

```cpp
#include "profiler.hpp"

int main() {
    // Setup code...

    g_profiler.start("total_execution");

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        PROFILE_SCOPE("epoch");

        // Existing training code...
    }

    g_profiler.stop("total_execution");
    g_profiler.print_report();

    return 0;
}
```

## Interpreting Results

The profiler will output a report showing:
- Total time per segment
- Number of calls to each segment
- Average time per call
- Percentage of total time

The primary bottleneck will be identified as the segment taking the most total time.

## Optimization Tips

Based on common bottlenecks in neural network training:

1. **Matrix Operations**: Consider BLAS libraries or vectorized operations
2. **Convolution**: Look into optimized convolution algorithms (FFT-based, Winograd)
3. **Memory Management**: Minimize allocations in the training loop
4. **Batch Sizes**: Experiment with different batch sizes
5. **Activation Functions**: Use simpler activation functions if possible

## Further Analysis

For deeper performance analysis, consider:
- Memory profiling tools
- Cache analysis tools
- Instruction-level profiling
