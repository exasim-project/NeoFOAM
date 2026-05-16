# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)

include(CTest)
# include(Catch)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

if(NEOFOAM_WITH_MPI)
  add_library(neofoam_catch_main_mpi test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                     mpiSerialization.cpp)
  target_link_libraries(neofoam_catch_main_mpi PUBLIC OpenFOAM Catch2::Catch2 NeoFOAM NeoN
                                                      MPI::MPI_CXX)
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_MPI_SUPPORT=1)
endif()

add_library(neofoam_catch_main test_main.cpp)
target_link_libraries(neofoam_catch_main PUBLIC OpenFOAM Catch2::Catch2 NeoFOAM)
install(TARGETS neofoam_catch_main Catch2)
