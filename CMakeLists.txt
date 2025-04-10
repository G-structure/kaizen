cmake_minimum_required(VERSION 3.10)
project(ProofOfLearning)

set(CMAKE_CXX_STANDARD 14)

# Check the architecture and set appropriate flags
if(DEFINED USE_ARM_ARCHITECTURE OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)"))
    message(STATUS "Building for ARM architecture")
    set(CMAKE_CXX_FLAGS "-w -O3 -faligned-new -lgmp")
    set(CMAKE_C_FLAGS "-w -O3 -faligned-new -lgmp")
    add_definitions(-DUSE_ARM_ARCHITECTURE)
else()
    message(STATUS "Building for x86 architecture")
    set(CMAKE_CXX_FLAGS "-w -march=native -faligned-new -mavx -lgmp -msha")
    set(CMAKE_C_FLAGS "-w -march=native -faligned-new -mavx -lgmp -msha")
    add_definitions(-DUSE_X86_INTRINSICS)
endif()

link_directories(3rd/mcl)
link_directories(lib)

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)
include_directories(lib)

add_subdirectory(src)
add_subdirectory(3rd/mcl)
