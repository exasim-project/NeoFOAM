# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

#!/usr/bin/env bash
set -euo pipefail

# Check required environment variables
GPU_VENDOR=${GPU_VENDOR:?Error: Must set GPU vendor (nvidia|amd|intel)}
NEON_BRANCH=${NEON_BRANCH:?Error: Must set NeoN branch}
PRESET="develop"

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
    export CXX_COMPILER_PATH="$(which g++)"
    export CXX_SOURCE="${CXX_COMPILER_PATH%/*/*}"
    export CXX_LIBDIR="${CXX_SOURCE}/lib64"
    export LD_LIBRARY_PATH=${CXX_LIBDIR}:${LD_LIBRARY_PATH}

    echo "=== AMD GPU info ==="
    rocminfo | grep "Marketing Name.*AMD"
    echo "=== AMD compiler driver info ==="
    hipcc --version

elif [[ "$GPU_VENDOR" == "intel" ]]; then
    SYCL_PI_TRACE=1 
    sycl-ls
    # Compiler info (non-fatal)
    icpx --version 2>/dev/null | head -1 || echo "icpx not found"

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
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DNeoN_WITH_THREADS=OFF \
        -DNEOFOAM_BUILD_BENCHMARKS=ON
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    cmake --preset $PRESET \
        -DNEOFOAM_NEON_DIR=../NeoN \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_CXX_FLAGS="--gcc-toolchain=${CXX_SOURCE}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CXX_LIBDIR}" \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DNeoN_WITH_THREADS=OFF \
        -DNEOFOAM_BUILD_BENCHMARKS=ON
elif [[ "$GPU_VENDOR" == "intel" ]]; then
    cmake --preset $PRESET \
        -DNEOFOAM_NEON_DIR=../NeoN \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat -ffp-model=precise" \
        -DKokkos_ENABLE_SYCL=ON \
        -DKokkos_ARCH_INTEL_PVC=ON \
        -DNeoN_WITH_THREADS=OFF \
        -DNEOFOAM_BUILD_BENCHMARKS=ON
fi

echo "=== Building NeoFOAM against NeoN ==="
cmake --build --preset $PRESET

# -------------------------
# Step 3: Run Tests
# -------------------------
echo "=== Running NeoFOAM tests ==="
ctest --preset $PRESET -R neofoam --output-on-failure

# -----------------------------
# Step 4: Validate neoIcoFoam
# -----------------------------
pushd tutorials/cavity >/dev/null
python3 cleanRunValidate.py --preset "$PRESET"
popd >/dev/null
