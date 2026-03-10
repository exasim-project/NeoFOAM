#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

PRESET="profiling"

# Check required environment variables
GPU_VENDOR=${GPU_VENDOR:?Error: Must set GPU vendor (nvidia|amd|intel)}
NEON_BRANCH=${NEON_BRANCH:?Error: Must set NeoN branch}
PR_NUMBER=${PR_NUMBER:?Error: Must set PR number}
RESULTS_DIR=${RESULTS_DIR:-results}
TARGET_REPO=${TARGET_REPO:?Must set TARGET_REPO}
REPO_NAME=$(basename "$TARGET_REPO" .git)
TARGET_BRANCH=${TARGET_BRANCH:?Must set TARGET_BRANCH}
RUN_IDENTIFIER=${RUN_IDENTIFIER:?Must set RUN_IDENTIFIER}
API_TOKEN_GITHUB=${API_TOKEN_GITHUB:?Must set API_TOKEN_GITHUB}

echo "Selected GPU vendor: ${GPU_VENDOR}"

# Collect system info
collect_system_info() {
    mkdir -p "$RESULTS_DIR"
    {
        echo "===== CPU INFO ====="
        lscpu || echo "lscpu not available"
        echo ""

        echo "===== GPU INFO ====="
        if [[ "$GPU_VENDOR" == "nvidia" ]]; then
            nvidia-smi
        elif [[ "$GPU_VENDOR" == "amd" ]]; then
            rocm-smi --showproductname --showvbios
        elif [[ "$GPU_VENDOR" == "intel" ]]; then
            if ! sycl-ls --ignore-device-selectors 2>/dev/null | grep -qi intel; then
                echo "No Intel GPU found or Level Zero runtime not available"
            fi
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

# -------------------------
# Step 2: Configure and build NeoFOAM for benchmarking
# -------------------------
build_and_benchmark() {
    local branch=$1
    local output_dir=$2
    export CTEST_OUTPUT_ON_FAILURE=1

    echo ">>> Checking out ${branch}"
    git fetch origin "${branch}"
    git checkout "${branch}"

    echo ">>> Configuring build"
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        cmake --preset $PRESET \
            -DNEOFOAM_NEON_DIR=../NeoN \
            -DCMAKE_CUDA_ARCHITECTURES=90 \
            -DNeoN_WITH_THREADS=ON
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
        # Set up environment
        export CXX_COMPILER_PATH="$(which g++)"
        export CXX_SOURCE="${CXX_COMPILER_PATH%/*/*}"
        export CXX_LIBDIR="${CXX_SOURCE}/lib64"
        export LD_LIBRARY_PATH=${CXX_LIBDIR}:${LD_LIBRARY_PATH}

        cmake --preset $PRESET \
            -DNEOFOAM_NEON_DIR=../NeoN \
            -DCMAKE_PREFIX_PATH=/opt/rocm \
            -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
            -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
            -DCMAKE_CXX_FLAGS="--gcc-toolchain=${CXX_SOURCE}" \
            -DCMAKE_EXE_LINKER_FLAGS="-L${CXX_LIBDIR}" \
            -DCMAKE_HIP_ARCHITECTURES=gfx90a \
            -DKokkos_ARCH_AMD_GFX90A=ON \
            -DNeoN_WITH_THREADS=ON
    elif [[ "$GPU_VENDOR" == "intel" ]]; then
        cmake --preset $PRESET \
        -DNEOFOAM_NEON_DIR=../NeoN \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat -ffp-model=precise" \
        -DKokkos_ENABLE_SYCL=ON \
        -DKokkos_ARCH_INTEL_PVC=ON \
        -DNeoN_WITH_THREADS=ON \
        -DNEOFOAM_BENCHMARK_MODE="fast" \
        -DCMAKE_BUILD_TYPE="release"
    else
        cmake --preset $PRESET -DNEOFOAM_NEON_DIR=../NeoN -DNeoN_WITH_THREADS=OFF
    fi

    echo ">>> Building"
    cmake --build --preset $PRESET
    echo ">>> Running benchmarks..."
    export PATH=$PATH:$PWD/build/$PRESET/bin/benchmarks
    if [[ "$GPU_VENDOR" == "intel" ]]; then
        export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    fi
    ctest --preset profiling
    echo ">>> Benchmarks completed"

    # Check for produced results
    find build  -name "results"  -exec python3 benchmarks/benchmarkSuite/createStudies.py display {} \; > results.md
    cat results.md
    mapfile -d '' csv_files < <(find build/profiling/benchmarkSuite/ -type f -name '*.csv' -print0)

    if [ "${#csv_files[@]}" -eq 0 ]; then
        echo "No CSV files found!" >&2
    fi

    # Display the list of files generated
    echo ">>> List of files generated."
    for f in "${csv_files[@]}"; do
        echo "$f"
    done
    echo "============================"

    # Copy the files to a common directory
    mkdir -p -- "${output_dir}"
    for f in "${csv_files[@]}"; do
        cp -f -- "$f" "${output_dir}/"
    done

    rm -rf build
}

# Push benchmark results to GitHub
push_results() {
    git clone "https://oauth2:${API_TOKEN_GITHUB}@${TARGET_REPO}"
    cd "${REPO_NAME}"

    git config user.email "gitlab-ci@users.noreply.github.com"
    git config user.name "GitLab CI"

    git checkout "NeoFOAM_PR_${PR_NUMBER}" || git checkout -b "NeoFOAM_PR_${PR_NUMBER}"
    mkdir -p "${RESULTS_DIR}"
    cp -r ../${RESULTS_DIR}/* "${RESULTS_DIR}"

    git add .
    git commit -m "Benchmarks from GitLab pipeline ${RUN_IDENTIFIER}" || echo "No changes to commit"
    git pull --rebase || true
    git push origin "NeoFOAM_PR_${PR_NUMBER}"
}

### Main execution ###
collect_system_info "${GPU_VENDOR}"

# Current branch
echo ">>> Benchmarking the current branch"
build_and_benchmark "$(git rev-parse --abbrev-ref HEAD)" "${RESULTS_DIR}"

# Push results
echo ">>> Copying results to NeoFOAM-BenchmarkData repository"
push_results
echo "Results copied successfully"
