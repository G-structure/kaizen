//
// Self-contained CNN profiler test
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
    F& operator+=(const F& other) { value += other.value; return *this; }
    
    double value;
};

// Helper function to compute convolution product
F conv_prod(const std::vector<std::vector<F>>& x, 
            const std::vector<std::vector<F>>& w,
            int ii, int jj) 
{
    PROFILE_SCOPE("conv_prod");
    F sum = F(0);
    for (size_t i = 0; i < w.size(); i++) {
        for (size_t j = 0; j < w[i].size(); j++) {
            sum += w[i][j] * x[i + ii][j + jj];
        }
    }
    return sum;
}

// Add matrices helper
std::vector<std::vector<F>> add_matrix(
    const std::vector<std::vector<F>>& M1, 
    const std::vector<std::vector<F>>& M2) 
{
    PROFILE_SCOPE("add_matrix");
    std::vector<std::vector<F>> M(M1.size());
    for (size_t i = 0; i < M1.size(); i++) {
        M[i].resize(M1[i].size());
        for (size_t j = 0; j < M1[i].size(); j++) {
            M[i][j] = M1[i][j] + M2[i][j];
        }
    }
    return M;
}

// Simple 2D convolution
std::vector<std::vector<F>> simple_convolution(
    const std::vector<std::vector<F>>& x, 
    const std::vector<std::vector<F>>& w) 
{
    PROFILE_SCOPE("simple_convolution");
    
    // Profile output allocation
    std::vector<std::vector<F>> out(x.size() - w.size() + 1);
    {
        PROFILE_SCOPE("output_allocation");
        for (size_t i = 0; i < out.size(); i++) {
            out[i].resize(x[0].size() - w[0].size() + 1);
        }
    }
    
    // Profile convolution computation
    {
        PROFILE_SCOPE("conv_loop");
        for (size_t i = 0; i < out.size(); i++) {
            for (size_t j = 0; j < out[i].size(); j++) {
                out[i][j] = conv_prod(x, w, i, j);
            }
        }
    }
    
    return out;
}

// Multi-channel convolution
std::vector<std::vector<std::vector<F>>> convolution_profiled(
    const std::vector<std::vector<std::vector<F>>>& input, 
    const std::vector<std::vector<std::vector<std::vector<F>>>>& w) 
{
    PROFILE_SCOPE("convolution");
    
    size_t ch_out = w.size();
    size_t ch_in = w[0].size();
    size_t W_dim = w[0][0].size();
    size_t n = input[0].size();
    size_t output_dim = n - W_dim + 1;
    
    // Profile memory allocation
    std::vector<std::vector<std::vector<F>>> conv(ch_out);
    {
        PROFILE_SCOPE("conv_memory_allocation");
        for (size_t i = 0; i < ch_out; i++) {
            conv[i].resize(output_dim);
            for (size_t j = 0; j < output_dim; j++) {
                conv[i][j].resize(output_dim, F(0));
            }
        }
    }
    
    // Profile computations
    {
        PROFILE_SCOPE("conv_computation");
        for (size_t i = 0; i < ch_out; i++) {
            for (size_t j = 0; j < ch_in; j++) {
                // Perform simple convolution for this input-output channel pair
                std::vector<std::vector<F>> temp_conv = simple_convolution(input[j], w[i][j]);
                
                // Add to output
                conv[i] = add_matrix(conv[i], temp_conv);
            }
        }
    }
    
    return conv;
}

// Helper function to compute average of a window
F compute_avg(const std::vector<std::vector<F>>& x, int i_start, int j_start, int dim) {
    PROFILE_SCOPE("compute_avg");
    F sum = F(0);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            sum += x[i + i_start][j + j_start];
        }
    }
    return F(sum.value / (dim * dim));
}

// Average pooling
std::vector<std::vector<std::vector<F>>> avg_pool_profiled(
    const std::vector<std::vector<std::vector<F>>>& input, 
    int filter_dim) 
{
    PROFILE_SCOPE("avg_pooling");
    
    size_t n = input[0].size();
    size_t ch = input.size();
    int stride = filter_dim;
    int output_dim = (n - filter_dim) / stride + 1;
    
    // Profile allocation
    std::vector<std::vector<std::vector<F>>> output(ch);
    {
        PROFILE_SCOPE("pool_allocation");
        for (size_t i = 0; i < ch; i++) {
            output[i].resize(output_dim);
            for (int j = 0; j < output_dim; j++) {
                output[i][j].resize(output_dim);
            }
        }
    }
    
    // Profile pooling computation
    {
        PROFILE_SCOPE("pool_computation");
        for (size_t c = 0; c < ch; c++) {
            for (int i = 0; i < output_dim; i++) {
                for (int j = 0; j < output_dim; j++) {
                    output[c][i][j] = compute_avg(input[c], i * stride, j * stride, filter_dim);
                }
            }
        }
    }
    
    return output;
}

// ReLU activation for 3D feature maps
std::vector<std::vector<std::vector<F>>> relu_3d(const std::vector<std::vector<std::vector<F>>>& input) {
    PROFILE_SCOPE("relu_3d");
    
    std::vector<std::vector<std::vector<F>>> output(input.size());
    for (size_t c = 0; c < input.size(); c++) {
        output[c].resize(input[c].size());
        for (size_t i = 0; i < input[c].size(); i++) {
            output[c][i].resize(input[c][i].size());
            for (size_t j = 0; j < input[c][i].size(); j++) {
                output[c][i][j] = (input[c][i][j].value < 0) ? F(0) : input[c][i][j];
            }
        }
    }
    
    return output;
}

// Simple test function
void run_cnn_profiling_test() {
    std::cout << "Running CNN profiling test...\n";
    
    // Reset profiler
    g_profiler.reset();
    
    // Overall timing
    g_profiler.start("total_cnn_execution");
    
    // Simulate CNN operations
    const int input_size = 28;  // 28x28 image
    const int kernel1_size = 5; // 5x5 kernel
    const int kernel2_size = 3; // 3x3 kernel
    const int pool_size = 2;    // 2x2 pooling
    const int num_channels = 3; // RGB
    const int num_filters1 = 8; // First conv layer filters
    const int num_filters2 = 16; // Second conv layer filters
    
    // Create simulated input image and kernels
    std::vector<std::vector<std::vector<F>>> input_image;
    std::vector<std::vector<std::vector<std::vector<F>>>> kernels1; // First layer kernels
    std::vector<std::vector<std::vector<std::vector<F>>>> kernels2; // Second layer kernels
    
    // Initialize with some dummy data
    {
        PROFILE_SCOPE("data_initialization");
        
        // Create input image
        input_image.resize(num_channels);
        for (int c = 0; c < num_channels; c++) {
            input_image[c].resize(input_size);
            for (int i = 0; i < input_size; i++) {
                input_image[c][i].resize(input_size);
                for (int j = 0; j < input_size; j++) {
                    input_image[c][i][j] = F(0.01 * (c + i + j));
                }
            }
        }
        
        // Create kernels for first layer
        kernels1.resize(num_filters1);
        for (int f = 0; f < num_filters1; f++) {
            kernels1[f].resize(num_channels);
            for (int c = 0; c < num_channels; c++) {
                kernels1[f][c].resize(kernel1_size);
                for (int i = 0; i < kernel1_size; i++) {
                    kernels1[f][c][i].resize(kernel1_size);
                    for (int j = 0; j < kernel1_size; j++) {
                        kernels1[f][c][i][j] = F(0.01 * (f + c - i - j));
                    }
                }
            }
        }
        
        // Create kernels for second layer
        kernels2.resize(num_filters2);
        for (int f = 0; f < num_filters2; f++) {
            kernels2[f].resize(num_filters1); // Input is output of first layer
            for (int c = 0; c < num_filters1; c++) {
                kernels2[f][c].resize(kernel2_size);
                for (int i = 0; i < kernel2_size; i++) {
                    kernels2[f][c][i].resize(kernel2_size);
                    for (int j = 0; j < kernel2_size; j++) {
                        kernels2[f][c][i][j] = F(0.01 * (f - c + i + j));
                    }
                }
            }
        }
    }
    
    // Run first convolution layer
    std::vector<std::vector<std::vector<F>>> conv1_output;
    {
        PROFILE_SCOPE("conv_layer1");
        conv1_output = convolution_profiled(input_image, kernels1);
        
        // Apply ReLU
        conv1_output = relu_3d(conv1_output);
    }
    
    // Run first pooling layer
    std::vector<std::vector<std::vector<F>>> pool1_output;
    {
        PROFILE_SCOPE("pool_layer1");
        pool1_output = avg_pool_profiled(conv1_output, pool_size);
    }
    
    // Run second convolution layer
    std::vector<std::vector<std::vector<F>>> conv2_output;
    {
        PROFILE_SCOPE("conv_layer2");
        conv2_output = convolution_profiled(pool1_output, kernels2);
        
        // Apply ReLU
        conv2_output = relu_3d(conv2_output);
    }
    
    // Run second pooling layer
    std::vector<std::vector<std::vector<F>>> pool2_output;
    {
        PROFILE_SCOPE("pool_layer2");
        pool2_output = avg_pool_profiled(conv2_output, pool_size);
    }
    
    // Print summary
    std::cout << "CNN forward pass completed." << std::endl;
    if (!pool2_output.empty() && !pool2_output[0].empty() && !pool2_output[0][0].empty()) {
        std::cout << "Final output sample: " << pool2_output[0][0][0].value << std::endl;
    }
    
    // Stop timing and print report
    g_profiler.stop("total_cnn_execution");
    g_profiler.print_report();
}

int main() {
    run_cnn_profiling_test();
    return 0;
} 