//
// Self-contained MLP profiler test
//

#include "profiler.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// For this simplified test, we'll define our own F type
class F {
public:
    F() : value(0.0) {}
    F(double v) : value(v) {}
    
    F operator+(const F& other) const { return F(value + other.value); }
    F operator-(const F& other) const { return F(value - other.value); }
    F operator*(const F& other) const { return F(value * other.value); }
    bool operator<(const F& other) const { return value < other.value; }
    
    double value;
};

// Helper functions for the test
F inner_product(const std::vector<F>& v1, const std::vector<F>& v2) {
    F sum = F(0);
    for (size_t i = 0; i < v1.size(); i++) {
        sum = sum + (v1[i] * v2[i]);
    }
    return sum;
}

// Matrix multiplication function with profiling
std::vector<std::vector<F>> matrix_mul_profiled(
    const std::vector<std::vector<F>>& M1, 
    const std::vector<std::vector<F>>& M2) 
{
    PROFILE_SCOPE("matrix_multiplication");
    
    std::vector<std::vector<F>> M;
    M.resize(M1.size());
    
    // Profile the memory allocation separately
    {
        PROFILE_SCOPE("matrix_memory_allocation");
        for(size_t i = 0; i < M1.size(); i++) {
            M[i].resize(M2.size());
        }
    }
    
    // Profile the actual computation
    {
        PROFILE_SCOPE("matrix_computation");
        for(size_t i = 0; i < M1.size(); i++) {
            for(size_t j = 0; j < M2.size(); j++) {
                M[i][j] = inner_product(M1[i], M2[j]);
            }
        }
    }
    
    return M;
}

// ReLU activation function with profiling
std::vector<std::vector<F>> relu_profiled(const std::vector<std::vector<F>>& M1) {
    PROFILE_SCOPE("relu_activation");
    
    std::vector<std::vector<F>> M;
    M.resize(M1.size());
    
    for(size_t i = 0; i < M1.size(); i++) {
        M[i].resize(M1[i].size());
        for(size_t j = 0; j < M1[i].size(); j++) {
            if(M1[i][j] < F(0)) {
                M[i][j] = F(0);
            } else {
                M[i][j] = M1[i][j];
            }
        }
    }
    
    return M;
}

// Add bias function with profiling
std::vector<std::vector<F>> add_bias_profiled(
    const std::vector<std::vector<F>>& M1, 
    const std::vector<F>& bias) 
{
    PROFILE_SCOPE("add_bias");
    
    std::vector<std::vector<F>> M = M1;
    
    for(size_t i = 0; i < M.size(); i++) {
        for(size_t j = 0; j < M[i].size(); j++) {
            M[i][j] = M[i][j] + bias[j];
        }
    }
    
    return M;
}

// Feedforward function with profiling
std::vector<std::vector<F>> feedforward_profiled(
    const std::vector<std::vector<F>>& input,
    const std::vector<std::vector<std::vector<F>>>& weights,
    const std::vector<std::vector<F>>& biases)
{
    PROFILE_SCOPE("feedforward");
    
    std::vector<std::vector<F>> current_output = input;
    
    for (size_t layer = 0; layer < weights.size(); layer++) {
        std::vector<std::vector<F>> layer_output;
        
        // Forward through this layer
        {
            PROFILE_SCOPE("layer_" + std::to_string(layer + 1));
            
            // Matrix multiplication
            layer_output = matrix_mul_profiled(current_output, weights[layer]);
            
            // Add bias
            layer_output = add_bias_profiled(layer_output, biases[layer]);
            
            // Apply ReLU activation (except for last layer)
            if (layer < weights.size() - 1) {
                layer_output = relu_profiled(layer_output);
            }
        }
        
        current_output = layer_output;
    }
    
    return current_output;
}

// Simple test function
void run_mlp_profiling_test() {
    std::cout << "Running MLP profiling test...\n";
    
    // Reset profiler
    g_profiler.reset();
    
    // Overall timing
    g_profiler.start("total_mlp_execution");
    
    // Simulate a small MLP operation
    const int input_size = 100;
    const int hidden_size = 50;
    const int output_size = 10;
    const int batch_size = 32;
    const int num_layers = 2;  // input->hidden->output
    
    // Create simulated data
    std::vector<std::vector<F>> input_data;
    std::vector<std::vector<std::vector<F>>> weights(num_layers);
    std::vector<std::vector<F>> biases(num_layers);
    
    // Initialize with some dummy data
    {
        PROFILE_SCOPE("data_initialization");
        
        // Create input data
        input_data.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            input_data[i].resize(input_size);
            for (int j = 0; j < input_size; j++) {
                input_data[i][j] = F(0.01 * (i + j));
            }
        }
        
        // Create first layer weights (input -> hidden)
        weights[0].resize(input_size);
        for (int i = 0; i < input_size; i++) {
            weights[0][i].resize(hidden_size);
            for (int j = 0; j < hidden_size; j++) {
                weights[0][i][j] = F(0.01 * (i - j));
            }
        }
        
        // Create first layer bias
        biases[0].resize(hidden_size);
        for (int j = 0; j < hidden_size; j++) {
            biases[0][j] = F(0.1);
        }
        
        // Create second layer weights (hidden -> output)
        weights[1].resize(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            weights[1][i].resize(output_size);
            for (int j = 0; j < output_size; j++) {
                weights[1][i][j] = F(0.02 * (i + j));
            }
        }
        
        // Create second layer bias
        biases[1].resize(output_size);
        for (int j = 0; j < output_size; j++) {
            biases[1][j] = F(0.2);
        }
    }
    
    // Run complete feedforward pass
    std::vector<std::vector<F>> output;
    {
        PROFILE_SCOPE("complete_mlp_forward");
        output = feedforward_profiled(input_data, weights, biases);
    }
    
    // Print a sample of the output
    std::cout << "Sample output: " << output[0][0].value << std::endl;
    
    // Stop timing and print report
    g_profiler.stop("total_mlp_execution");
    g_profiler.print_report();
}

int main() {
    run_mlp_profiling_test();
    return 0;
} 