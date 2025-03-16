#!/bin/bash

echo "Building Kaizen for ARM architecture..."

# Make sure 3rd party dependencies are extracted
if [ ! -d "3rd" ]; then
    echo "Extracting 3rd party dependencies..."
    unzip 3rd.zip
fi

# Create build directory
mkdir -p build_arm
cd build_arm

# Set up environment variables for GMP
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/Cellar/gmp/6.3.0/lib
export CPATH=$CPATH:/opt/homebrew/Cellar/gmp/6.3.0/include

# Configure with ARM-specific flags and Clang as required by MCL
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_ARM_ARCHITECTURE=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DUSESHA3=0 ..

# Build
make

echo "Build completed. You can now run the test scripts."
cd .. 