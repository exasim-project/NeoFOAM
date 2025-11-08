# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

#!/usr/bin/env bash
set -euo pipefail

# Argument parsing
GPU_VENDOR=${1:?Error: GPU vendor (nvidia|amd) must be specified}
NEON_BRANCH=${2:-main} # Default to 'main' if not provided
PRESET=${3:-develop}   # Which CMakePreset to use, defaults to 'develop'

echo "=== GPU vendor=$GPU_VENDOR, NeoN branch=$NEON_BRANCH ==="
# -------------------------
# Step 0: GPU/Compiler/Tool Info
# -------------------------
echo "=== Tool versions ==="
cmake --version
g++ --version || clang++ --version

if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    echo "=== NVIDIA GPU info ==="
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
    echo "=== NVIDIA compiler driver info ==="
    nvcc --version

elif [[ "$GPU_VENDOR" == "amd" ]]; then
    # Set up environment
    export PATH=/opt/rocm/bin:$PATH
    export HIPCC_CXX=/usr/bin/g++

    echo "=== AMD GPU info ==="
    rocminfo | grep "Marketing Name.*AMD"
    echo "=== AMD compiler driver info ==="
    hipcc --version
else
    echo "Unsupported GPU vendor: $GPU_VENDOR"
    exit 1
fi

# -------------------------
# Step 1: Prepare NeoN
# -------------------------
echo "=== Cloning NeoN (branch=$NEON_BRANCH) ==="
git clone --depth 1 --single-branch --branch "$NEON_BRANCH" \
    https://gitlab-ce.lrz.de/greole/neon.git ../NeoN

# -------------------------
# Step 2: Configure and build NeoFOAM
# -------------------------
echo "=== Configuring NeoFOAM against NeoN ==="

if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    cmake --preset $PRESET \
        -DNEOFOAM_NEON_DIR=../NeoN \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DNeoN_WITH_THREADS=OFF
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    cmake --preset $PRESET \
        -DNEOFOAM_NEON_DIR=../NeoN \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DNeoN_WITH_THREADS=OFF
fi

echo "=== Building NeoFOAM against NeoN ==="
cmake --build --preset $PRESET

# -------------------------
# Step 3: Run Tests
# -------------------------
echo "=== Running NeoFOAM tests ==="
ctest --preset $PRESET -R adapter --output-on-failure
