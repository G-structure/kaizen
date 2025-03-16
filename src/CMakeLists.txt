# Enable CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

# Make sure CUDA is used
enable_language(CUDA)

# Add source files
aux_source_directory(. SRC)
include_directories(.)

# Define the CUDA source file
set(CUDA_SOURCES
    cuda_kernels.cu
)

# Remove any potential conflicting files from regular sources
list(REMOVE_ITEM SRC "cutlass_wrapper.cpp")

# Add the main executable
add_executable(main ${SRC} ${CUDA_SOURCES})

# Set CUDA architecture for this target
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES 70)

# Link against necessary libraries
target_link_libraries(main mcl mclbn384_256 gmp XKCP ${CUDA_LIBRARIES})
