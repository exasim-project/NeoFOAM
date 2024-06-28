# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

target_sources(NeoFOAM PRIVATE "time.cpp" "dictionary.cpp" "kokkos.cpp" "executor/OMPExecutor.cpp"
                               "executor/GPUExecutor.cpp" "executor/CPUExecutor.cpp")
