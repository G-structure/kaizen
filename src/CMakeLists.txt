aux_source_directory(. SRC)
include_directories(.)
add_executable(main ${SRC})

# Conditionally link against XKCP only for x86 architecture
if(DEFINED USE_ARM_ARCHITECTURE OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)"))
    target_link_libraries(main mcl mclbn384_256 gmp)
else()
    target_link_libraries(main mcl mclbn384_256 gmp XKCP)
endif()
