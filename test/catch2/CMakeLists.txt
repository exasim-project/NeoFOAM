# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

set(NEOFOAM_CATCH_MAIN "neofoam_catch_main")
set(NEOFOAM_CATCH_MAIN_MPI "${NEOFOAM_CATCH_MAIN}_mpi")
set(NEOFOAM_CATCH_MAIN_MPI_SRCS test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                mpiSerialization.cpp)
add_library(neofoam_catch_main_mpi ${NEOFOAM_CATCH_MAIN_MPI_SRCS})
if(Kokkos_ENABLE_CUDA)
  set_source_files_properties(${NEOFOAM_CATCH_MAIN_MPI_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

target_link_libraries(
  ${NEOFOAM_CATCH_MAIN_MPI} PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos MPI::MPI_C
                                   MPI::MPI_CXX NeoFOAM)
target_link_libraries(${NEOFOAM_CATCH_MAIN_MPI} PRIVATE neofoam_warnings neofoam_options)

set(NEOFOAM_CATCH_MAIN_SRCS test_main.cpp)
add_library(${NEOFOAM_CATCH_MAIN} ${NEOFOAM_CATCH_MAIN_SRCS})
if(Kokkos_ENABLE_CUDA)
  set_source_files_properties(${NEOFOAM_CATCH_MAIN_SRCS} PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(${NEOFOAM_CATCH_MAIN} PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos
                                                   NeoFOAM)
target_link_libraries(${NEOFOAM_CATCH_MAIN} PRIVATE neofoam_warnings neofoam_options)
