#!/bin/bash

echo "=== Kaizen GPU Runner ==="
echo "This script will help run Kaizen with GPU acceleration"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo "Warning: NVIDIA Container Toolkit might not be properly installed"
    echo "GPU acceleration may not work correctly"
fi

# Build the Docker container
echo "Building CUDA-enabled Docker container for Kaizen..."
docker build -t kaizen-gpu -f Dockerfile.gpu .

# Menu for options
while true; do
    echo ""
    echo "Select an option:"
    echo "1. Run GPU test (verify GPU functionality)"
    echo "2. Run AlexNet tests"
    echo "3. Run VGG tests"
    echo "4. Run LeNet tests"
    echo "5. Start interactive shell"
    echo "6. Exit"
    read -p "Enter your choice [1-6]: " choice
    
    case $choice in
        1)
            echo "Running GPU test..."
            docker run --gpus all -it kaizen-gpu ./test_gpu
            ;;
        2)
            echo "Running AlexNet tests..."
            docker run --gpus all -it kaizen-gpu ./alexnet_test_gpu.sh
            ;;
        3)
            echo "Running VGG tests..."
            docker run --gpus all -it kaizen-gpu ./vgg_gpu.sh
            ;;
        4)
            echo "Running LeNet tests..."
            docker run --gpus all -it kaizen-gpu ./lenet_test_gpu.sh
            ;;
        5)
            echo "Starting interactive shell..."
            docker run --gpus all -it kaizen-gpu /bin/bash
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done 