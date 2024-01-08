# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

option(FORCE_NEOFOAM_BUILD_KOKKOS
       "Forces NeoFOAM to build kokkos even if kokkos is already present on the system" FALSE)

set(KOKKOS_CHECKOUT_VERSION
    "4.2.00"
    CACHE STRING "Use specific version of Kokkos")

set(KOKKOS_SOURCE
 "https://github.com/kokkos/kokkos.git"
    CACHE STRING "Use specific source of Kokkos")

find_package(Kokkos ${KOKKOS_CHECKOUT_VERSION} QUIET)

if (NOT ${Kokkos_FOUND} OR ${FORCE_NEOFOAM_BUILD_KOKKOS})
	if (${NEOFOAM_CUDA}) 
	    enable_language(CUDA)
	    find_package(CUDAToolkit REQUIRED)
	endif()


    include(ExternalProject)
    ExternalProject_Add(kokkos_external
      GIT_REPOSITORY ${KOKKOS_SOURCE}
      GIT_TAG ${KOKKOS_CHECKOUT_VERSION}
      GIT_SHALLOW ON
      CMAKE_ARGS
         -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install/kokkos
	 -DCMAKE_CXX_STANDARD=17
	 "-DKokkos_ENABLE_CUDA=${NEOFOAM_CUDA}"
	 "-DKokkos_ENABLE_OPENMP=${NEOFOAM_OMP}"
    )


    SET(Kokkos_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/install/kokkos/include)
    SET(Kokkos_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/install/lib)

    # Create the kokkos include dir now and populate it later
    file(MAKE_DIRECTORY ${Kokkos_INCLUDE_DIR})

    add_library(kokkos SHARED IMPORTED)
    set_property(TARGET kokkos PROPERTY
	                 IMPORTED_LOCATION "/path/to/libfoo.a")
    add_library(Kokkos::kokkos ALIAS kokkos)

    target_include_directories(kokkos INTERFACE
	    $<BUILD_INTERFACE:${Kokkos_INCLUDE_DIR}>
	    $<INSTALL_INTERFACE:${Kokkos_INCLUDE_DIR}>
	    )

endif()
