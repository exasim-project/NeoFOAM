# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

message(STATUS "Auto detecting accelerator devices")
include(CheckLanguage)

if(NOT DEFINED Kokkos_ENABLE_OPENMP AND NOT DEFINED Kokkos_ENABLE_THREADS)
  find_package(Threads QUIET)

  if(Threads_FOUND)
    message(STATUS "Set Kokkos_ENABLE_Threads=ON")
    set(Kokkos_ENABLE_THREADS
        ON
        CACHE INTERNAL "")
  else()
    find_package(OpenMP QUIET)

    if(OpenMP_FOUND)
      message(STATUS "Set Kokkos_ENABLE_OPENMP=ON")
      set(Kokkos_ENABLE_OPENMP
          ON
          CACHE INTERNAL "")
    endif()
  endif()
endif()

if(NOT DEFINED Kokkos_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(NEOFOAM_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
    message(STATUS "Set Kokkos_ENABLE_CUDA=ON")
    set(Kokkos_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR
        ON
        CACHE INTERNAL "")
  else()
    set(NEOFOAM_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")

    set(Kokkos_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip CUDA detection Kokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}")
endif()

if(NOT DEFINED Kokkos_ENABLE_HIP)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    message(STATUS "Set Kokkos_ENABLE_HIP=ON")
    set(Kokkos_ENABLE_HIP
        ON
        CACHE INTERNAL "")
  else()
    set(Kokkos_ENABLE_HIP
        OFF
        CACHE INTERNAL "")
  endif()
else()
  message(STATUS "Skip HIP detection Kokkos_ENABLE_HIP=${Kokkos_ENABLE_HIP}")
endif()

if(NOT DEFINED Kokkos_ENABLE_SERIAL)
  set(Kokkos_ENABLE_SERIAL
      ON
      CACHE INTERNAL "")
endif()
