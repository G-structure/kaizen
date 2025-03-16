//
// Example of using the profiler to identify bottlenecks in training code
//

#include "profiler.hpp"
#include "CNN.h"
#include "MLP.h"
#include <iostream>

// This function demonstrates how to use the profiler on existing code
void profile_training() {
    std::cout << "Starting profiled training...\n";
    
    // Initialize the network - profile this operation
    g_profiler.start("network_initialization");
    // Call your network initialization function here
    int model_init_result = initialize_model();
    g_profiler.stop("network_initialization");
    
    // Example loop for training
    const int num_epochs = 5;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Epoch " << (epoch+1) << "/" << num_epochs << std::endl;
        
        // Profile the feed forward phase
        {
            PROFILE_SCOPE("feed_forward");
            // Call your feedforward function here
            // For example: feedforward_transcript();
        }
        
        // Profile the backward propagation phase
        {
            PROFILE_SCOPE("back_propagation");
            // Call your backpropagation function here
            // For example: back_propagation_transcript();
        }
        
        // Profile weight updates
        g_profiler.start("weight_updates");
        // Call your weight update function here
        g_profiler.stop("weight_updates");
    }
    
    // Print the profiling report
    g_profiler.print_report();
}

// To use this example, add the following to your main.cpp:
/*
#include "profiler_example.cpp"

int main() {
    // Existing code...
    
    // Add this to profile your training
    profile_training();
    
    // Rest of your code...
    return 0;
}
*/

// Alternatively, you can directly modify your existing training code
// to add profiling instrumentation in the key sections:

/* Example of instrumenting existing code:

void your_existing_function() {
    PROFILE_SCOPE("your_existing_function");
    
    // Profile a specific section
    g_profiler.start("critical_computation");
    // Your critical computation code...
    g_profiler.stop("critical_computation");
    
    // Continue with other code...
}

*/ 