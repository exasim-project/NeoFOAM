# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(KOKKOS_CHECKOUT_VERSION
    "4.2.00"
    CACHE STRING "Use specific version of Kokkos")

find_package(Kokkos ${KOKKOS_CHECKOUT_VERSION} QUIET)

if (NOT ${Kokkos_FOUND} OR ${FORCE_NEOFOAM_BUILD_KOKKOS})
    include(FetchContent)

    FetchContent_Declare(
      kokkos
      QUITE
      GIT_SHALLOW ON
      GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
      GIT_TAG ${KOKKOS_CHECKOUT_VERSION}
    )

    FetchContent_MakeAvailable(Kokkos)
endif()
