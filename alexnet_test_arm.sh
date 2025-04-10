#!/bin/bash
# ARM version of alexnet_test.sh

# Rebuild with ARM optimizations
#echo "Rebuilding for ARM architecture..."
# ./build_arm.sh

echo "Running AlexNet tests on ARM..."
cd build_arm/src
./main mAlexNet 4 1 4 1
./main mAlexNet 8 1 4 1
./main mAlexNet 16 1 4 1
cd ../..
