#!/bin/bash
# ARM version of vgg.sh

# Rebuild with ARM optimizations
echo "Rebuilding for ARM architecture..."
./build_arm.sh

echo "Running VGG tests on ARM..."
cd build_arm/src
./main VGG 4 1 4 1
./main VGG 8 1 4 1
./main VGG 16 1 4 1
cd ../.. 