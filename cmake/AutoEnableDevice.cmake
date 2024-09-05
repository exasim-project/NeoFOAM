# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

include(CheckLanguage)

if(NOT DEFINED Kokkos_ENABLE_OPENMP AND NOT DEFINED Kokkos_ENABLE_THREADS)
  find_package(OpenMP QUIET)

  if(OpenMP_FOUND)
    # Check if the compiler is Clang
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      message(STATUS "Using Clang compiler")
      set(OpenMP_CXX_FLAGS "-fopenmp=libgomp")
      set(OpenMP_CXX_LIB_NAMES "gomp")
      set(OpenMP_gomp_LIBRARY gomp)
    endif()
    set(Kokkos_ENABLE_OPENMP
        ON
        CACHE INTERNAL "")
  else()
    find_package(Threads QUIET)

    if(Threads_FOUND)
      set(Kokkos_ENABLE_THREADS
          ON
          CACHE INTERNAL "")
    endif()
  endif()
endif()

if(NOT DEFINED Kokkos_ENABLE_CUDA)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    set(Kokkos_ENABLE_CUDA
        ON
        CACHE INTERNAL "")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR
        ON
        CACHE INTERNAL "")
  else()
    set(Kokkos_ENABLE_CUDA
        OFF
        CACHE INTERNAL "")
  endif()
endif()

if(NOT DEFINED Kokkos_ENABLE_HIP)
  check_language(HIP)

  if(CMAKE_HIP_COMPILER)
    set(Kokkos_ENABLE_HIP
        ON
        CACHE INTERNAL "")
  else()
    set(Kokkos_ENABLE_HIP
        OFF
        CACHE INTERNAL "")
  endif()
endif()
