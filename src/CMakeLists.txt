aux_source_directory(. SRC)
include_directories(.)

# Add CUDA sources conditionally
if(USE_CUDA)
    # Find CUDA if not already found
    find_package(CUDA REQUIRED)
    
    # Set the list of CUDA source files
    set(CUDA_SOURCES
        cuda_kernels.cu
    )
    
    # Compile CUDA files
    cuda_compile(CUDA_OBJECTS ${CUDA_SOURCES})
    
    # Add the executable with CUDA objects
    add_executable(main ${SRC} ${CUDA_OBJECTS})
    
    # Add CUDA libraries
    target_link_libraries(main ${CUDA_LIBRARIES})
else()
    # Regular non-CUDA build
    add_executable(main ${SRC})
endif()

target_link_libraries(main mcl mclbn384_256 gmp XKCP)
