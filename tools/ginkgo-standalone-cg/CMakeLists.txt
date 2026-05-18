# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Standalone Ginkgo distributed CG reproducer for the GPU multi-rank
# proc-boundary divergence bug — see .planning/debug/gpu-proc-boundary-divergence.md
# track T4.
#
# Reads a per-rank dump tree produced by an instrumented NeoFOAM run
# (NEOFOAM_FULL_DUMP=1) and replays the *exact* assembled distributed
# LinearSystem through Ginkgo's CG solver on both the Reference (CPU)
# executor and the Cuda executor. Diffs the two solutions to discriminate:
#
#   - L∞(x_cpu - x_cuda) ≈ 0     → Ginkgo is innocent; the NeoN→Ginkgo
#                                  composition layer (stream sharing / fence
#                                  coverage) is responsible.
#   - L∞(x_cpu - x_cuda) ≫ ε     → real Ginkgo distributed CUDA bug;
#                                  package as upstream reproducer.
#
# Build & run instructions are in this directory's README.md.

cmake_minimum_required(VERSION 3.22)
project(ginkgo_standalone_cg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# --- Dependencies --------------------------------------------------------- #
find_package(MPI 3.1 REQUIRED COMPONENTS CXX)

# Ginkgo must be the same version NeoN uses (see src/NeoN/cmake/Versions.cmake).
# Resolve from CMAKE_PREFIX_PATH on HPC; CI / dev machines pin to 2.0.0.
find_package(Ginkgo 2.0.0 CONFIG REQUIRED)

# --- Target --------------------------------------------------------------- #
add_executable(standalone_cg main.cpp)

target_link_libraries(standalone_cg
    PRIVATE
        Ginkgo::ginkgo
        MPI::MPI_CXX)

target_compile_definitions(standalone_cg PRIVATE OMPI_SKIP_MPICXX)

install(TARGETS standalone_cg RUNTIME DESTINATION bin)
