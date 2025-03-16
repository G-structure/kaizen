#!/bin/bash

# Check if GCC is installed
if command -v g++-13 &> /dev/null; then
    echo "GCC 13 is already installed."
else
    echo "GCC 13 is not installed. Installing via Homebrew..."
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        brew install gcc
    else
        echo "Error: Homebrew is not installed. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# Verify installation
g++-13 --version

echo "GCC is ready for use. You can now run ./build_arm.sh to build the project." 