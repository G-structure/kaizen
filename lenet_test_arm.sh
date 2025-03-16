#!/bin/bash
# ARM version of lenet_test.sh

# Rebuild with ARM optimizations
echo "Rebuilding for ARM architecture..."
./build_arm.sh

echo "Running LeNet tests on ARM..."
cd build_arm/src
./main LENET 1 1 -1 1
./main LENET 2 1 -1 1
./main LENET 4 1 1 1
./main LENET 8 1 1 1
./main LENET 16 1 2 1
cd ../.. 