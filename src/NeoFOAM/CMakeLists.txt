# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM)

include(GNUInstallDirs)

target_include_directories(
  NeoFOAM PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                 "$<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>")

target_sources(NeoFOAM
       	PRIVATE
       	"core/Time.cpp"
       	"core/Error.cpp"
       	"core/Kokkos.cpp"
	)

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(
  NeoFOAM ENABLE_NEOFOAM_SANITIZE_ADDRESS ENABLE_NEOFOAM_SANITIZE_LEAK
  ENABLE_NEOFOAM_SANITIZE_UB ENABLE_NEOFOAM_SANITIZE_THREAD
  ENABLE_NEOFOAM_SANITIZE_MEMORY)

target_link_libraries(NeoFOAM PRIVATE Kokkos::kokkos)

if (${NEOFOAM_CUDA}) 
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	target_include_directories(NeoFOAM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
