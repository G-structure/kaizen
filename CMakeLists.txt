cmake_minimum_required(VERSION 3.10)
project(ProofOfLearning LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w -march=native -faligned-new -mavx -lgmp -msha")
set(CMAKE_C_FLAGS "-w -march=native -faligned-new -mavx -lgmp -msha")

# Set CUDA architectures (this fixes the CMP0104 warning)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "-arch=sm_70")

# Enable CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include CUTLASS
include_directories(/root/projects/cutlass/include)
include_directories(/root/projects/cutlass/tools/util/include)

link_directories(3rd/mcl)
link_directories(lib)

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)
include_directories(lib)

add_subdirectory(src)
add_subdirectory(3rd/mcl)
