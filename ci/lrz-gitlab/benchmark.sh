#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

# Argument parsing
GPU_VENDOR=${1:?Error: GPU vendor (nvidia|amd) must be specified}
NEON_BRANCH=${2:-main} # Default to 'main' if not provided

RESULTS_DIR=${RESULTS_DIR:-results}
TARGET_REPO=${TARGET_REPO:?Must set TARGET_REPO}
REPO_NAME=$(basename "$TARGET_REPO" .git)
TARGET_BRANCH=${TARGET_BRANCH:?Must set TARGET_BRANCH}
RUN_IDENTIFIER=${RUN_IDENTIFIER:?Must set RUN_IDENTIFIER}
API_TOKEN_GITHUB=${API_TOKEN_GITHUB:?Must set API_TOKEN_GITHUB}

GPU_VENDOR="$1"
echo "Selected GPU vendor: ${GPU_VENDOR}"

# Collect system info
collect_system_info() {
    mkdir -p "$RESULTS_DIR"
    {
        echo "===== CPU INFO ====="
        lscpu || echo "lscpu not available"
        echo ""

        echo "===== GPU INFO ====="
        if [[ "$1" == "nvidia" ]]; then
            nvidia-smi
        elif [[ "$1" == "amd" ]]; then
            rocm-smi --showproductname --showvbios
        else
            echo "No GPU selected"
        fi
        echo ""

        echo "===== COMPILER INFO ====="
        echo "CMake:"
        cmake --version || echo "cmake not available"
        echo ""
        echo "C++ compiler:"
        g++ --version || clang++ --version || echo "No C++ compiler found"
        echo ""
        echo "CUDA/ROCm compiler:"
        nvcc --version 2>/dev/null || hipcc --version 2>/dev/null || echo "No GPU compiler available"
    } > "${RESULTS_DIR}/system-info.log"
}

# -------------------------
# Step 1: Prepare NeoN
# -------------------------
echo "=== Cloning NeoN (branch=$NEON_BRANCH) ==="
git clone --depth 1 --single-branch --branch "$NEON_BRANCH" \
    https://gitlab-ce.lrz.de/greole/neon.git ../NeoN
# Temporarly using NeoN from a previous state
current_dir=$(pwd)
cd ../NeoN
git fetch --unshallow
git checkout 283163928b803fe8778dfb214fa41430ea9f3ba6
cd $current_dir
# ============================================

# -------------------------
# Step 2: Configure and build FoamAdapter for benchmarking
# -------------------------
build_and_benchmark() {
    local branch=$1
    local output_dir=$2

    echo ">>> Checking out ${branch}"
    git fetch origin "${branch}"
    git checkout "${branch}"

    echo ">>> Configuring build"
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        cmake --preset profiling \
        -DFOAMADAPTER_NEON_DIR=../NeoN \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DNeoN_WITH_THREADS=OFF
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
        # Set up environment
        export PATH=/opt/rocm/bin:$PATH
        export HIPCC_CXX=/usr/bin/g++

        cmake --preset profiling \
        -DFOAMADAPTER_NEON_DIR=../NeoN \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DNeoN_WITH_THREADS=OFF
    else
        cmake --preset profiling -DFOAMADAPTER_NEON_DIR=../NeoN -DNeoN_WITH_THREADS=OFF
    fi

    echo ">>> Building"
    cmake --build --preset profiling

    echo ">>> Running benchmarks..."
    ./benchmarks/benchmarkSuite/cleanAll.sh
    ./benchmarks/benchmarkSuite/runAll.sh
    echo ">>> Benchmarks completed"

    # Check for produced results
    files=$(find benchmarks/benchmarkSuite/ -name '*csv')
    if [ -z "$files" ]; then
        echo "No CSV files found!"
        exit 1
    else
        echo ">>> List of files generated."
        echo "$files"
        echo "============================"

        mkdir -p "${output_dir}"
        cp $files "${output_dir}" \;
    fi
    
    rm -rf build
}

# Push benchmark results to GitHub
push_results() {
    git clone "https://oauth2:${API_TOKEN_GITHUB}@${TARGET_REPO}"
    cd "${REPO_NAME}"

    git config user.email "gitlab-ci@users.noreply.github.com"
    git config user.name "GitLab CI"

    git checkout "${TARGET_BRANCH}" || git checkout -b "${TARGET_BRANCH}"
    mkdir -p "${RESULTS_DIR}"
    cp -r ../${RESULTS_DIR}/* "${RESULTS_DIR}"

    git add .
    git commit -m "Benchmarks from GitLab pipeline ${RUN_IDENTIFIER}" || echo "No changes to commit"
    git pull --rebase || true
    git push origin "${TARGET_BRANCH}"
}

### Main execution ###
collect_system_info "${GPU_VENDOR}"

# Current branch
echo ">>> Benchmarking the current branch"
build_and_benchmark "$(git rev-parse --abbrev-ref HEAD)" "${RESULTS_DIR}"

# Main branch
echo ">>> Benchmarking the main branch"
build_and_benchmark "main" "${RESULTS_DIR}/main"

# Push results
echo ">>> Copying results to NeoFOAM-BenchmarkData repository"
push_results
echo "Results copied successfully"
