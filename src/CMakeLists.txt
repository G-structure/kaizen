aux_source_directory(. SRC)
include_directories(.)
add_executable(main ${SRC})

# Conditionally link against XKCP only for x86 architecture
if(DEFINED USE_ARM_ARCHITECTURE OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)"))
    target_link_libraries(main mcl mclbn384_256 gmp)
else()
    target_link_libraries(main mcl mclbn384_256 gmp XKCP)
endif()

# Add test executable for profiler
add_executable(run_profiler_test run_profiler_test.cpp profiler.cpp timer.cpp)

# Add MLP profiling test
add_executable(run_mlp_profiler run_mlp_profiler.cpp profiler.cpp timer.cpp)

# Add CNN profiling test
add_executable(run_cnn_profiler run_cnn_profiler.cpp profiler.cpp timer.cpp)
