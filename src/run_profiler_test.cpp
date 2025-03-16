//
// Simple test to demonstrate the profiler functionality
//

#include "profiler.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>

// Simulate a computationally intensive operation
void simulate_matrix_multiplication(int size) {
    PROFILE_SCOPE("matrix_multiplication");
    
    // Allocate matrices
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 1.0));
    std::vector<std::vector<double>> B(size, std::vector<double>(size, 2.0));
    std::vector<std::vector<double>> C(size, std::vector<double>(size, 0.0));
    
    // Perform matrix multiplication
    {
        PROFILE_SCOPE("multiplication_loop");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

// Simulate a convolution operation
void simulate_convolution(int input_size, int kernel_size) {
    PROFILE_SCOPE("convolution");
    
    // Allocate input and kernel
    std::vector<std::vector<double>> input(input_size, std::vector<double>(input_size, 1.0));
    std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size, 0.5));
    std::vector<std::vector<double>> output(input_size - kernel_size + 1, 
                                           std::vector<double>(input_size - kernel_size + 1, 0.0));
    
    // Perform convolution
    {
        PROFILE_SCOPE("convolution_loop");
        for (int i = 0; i < output.size(); i++) {
            for (int j = 0; j < output[0].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < kernel_size; k++) {
                    for (int l = 0; l < kernel_size; l++) {
                        sum += input[i + k][j + l] * kernel[k][l];
                    }
                }
                output[i][j] = sum;
            }
        }
    }
}

// Simulate gradient computation
void simulate_gradient_computation(int size) {
    PROFILE_SCOPE("gradient_computation");
    
    std::vector<double> weights(size, 0.5);
    std::vector<double> inputs(size, 1.0);
    std::vector<double> gradients(size, 0.0);
    
    // Compute some fake gradients
    for (int i = 0; i < size; i++) {
        double temp = 0.0;
        for (int j = 0; j < size; j++) {
            temp += inputs[j] * weights[(i + j) % size];
        }
        gradients[i] = std::sin(temp) * 0.01;
    }
    
    // Apply gradients
    {
        PROFILE_SCOPE("weight_update");
        for (int i = 0; i < size; i++) {
            weights[i] -= gradients[i];
        }
    }
}

// Simulate a forward pass
void simulate_forward_pass(int batch_size, int input_size, int hidden_size, int output_size) {
    PROFILE_SCOPE("forward_pass");
    
    // First layer
    {
        PROFILE_SCOPE("layer1_forward");
        simulate_matrix_multiplication(input_size);
        // Add activation simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Second layer
    {
        PROFILE_SCOPE("layer2_forward");
        simulate_matrix_multiplication(hidden_size);
        // Add activation simulation
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    // Output layer
    {
        PROFILE_SCOPE("output_layer_forward");
        simulate_matrix_multiplication(output_size);
    }
}

// Simulate a backward pass
void simulate_backward_pass(int batch_size, int input_size, int hidden_size, int output_size) {
    PROFILE_SCOPE("backward_pass");
    
    // Output layer gradients
    {
        PROFILE_SCOPE("output_layer_backward");
        simulate_gradient_computation(output_size);
    }
    
    // Hidden layer gradients
    {
        PROFILE_SCOPE("hidden_layer_backward");
        simulate_gradient_computation(hidden_size);
    }
    
    // Input layer gradients
    {
        PROFILE_SCOPE("input_layer_backward");
        simulate_gradient_computation(input_size);
    }
}

// Simulate a convolution neural network
void simulate_cnn() {
    PROFILE_SCOPE("cnn_forward");
    
    // Conv layer 1
    {
        PROFILE_SCOPE("conv_layer1");
        simulate_convolution(28, 5);
    }
    
    // Pooling layer 1
    {
        PROFILE_SCOPE("pooling_layer1");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    // Conv layer 2
    {
        PROFILE_SCOPE("conv_layer2");
        simulate_convolution(12, 3);
    }
    
    // Fully connected layers
    {
        PROFILE_SCOPE("fully_connected");
        simulate_matrix_multiplication(100);
    }
}

int main() {
    std::cout << "Running profiler test...\n";
    
    // Reset profiler state
    g_profiler.reset();
    
    // Start overall timing
    g_profiler.start("total_execution");
    
    // Simulate a training loop
    const int num_epochs = 2;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        PROFILE_SCOPE("epoch");
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        
        // Simulate batch processing
        const int num_batches = 3;
        for (int batch = 0; batch < num_batches; batch++) {
            PROFILE_SCOPE("batch_processing");
            
            // Simulate MLP training
            {
                PROFILE_SCOPE("mlp_training");
                simulate_forward_pass(32, 784, 128, 10);
                simulate_backward_pass(32, 784, 128, 10);
            }
            
            // Simulate CNN training
            {
                PROFILE_SCOPE("cnn_training");
                simulate_cnn();
                // CNN backward pass
                {
                    PROFILE_SCOPE("cnn_backward");
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
            }
        }
    }
    
    // Stop timing and print report
    g_profiler.stop("total_execution");
    g_profiler.print_report();
    
    return 0;
} 