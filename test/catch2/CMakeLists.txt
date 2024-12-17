# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  add_library(neofoam_catch_main_mpi test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                     mpiSerialization.cpp)
  if(NEOFOAM_ENABLE_CUDA)
    set_source_files_properties(test_main_mpi.cpp mpiReporter.cpp mpiGlobals.cpp
                                mpiSerialization.cpp PROPERTIES LANGUAGE CUDA)
  endif()
  target_link_libraries(neofoam_catch_main_mpi PUBLIC Catch2::Catch2 cpptrace::cpptrace
                                                      Kokkos::kokkos NeoFOAM_public_api)
  target_link_libraries(neofoam_catch_main_mpi PRIVATE neofoam_warnings neofoam_options)
endif()

add_library(neofoam_catch_main test_main.cpp)
if(NEOFOAM_ENABLE_CUDA)
  set_source_files_properties(test_main.cpp PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(neofoam_catch_main PUBLIC Catch2::Catch2 cpptrace::cpptrace Kokkos::kokkos
                                                NeoFOAM)
target_link_libraries(neofoam_catch_main PRIVATE neofoam_warnings neofoam_options)
