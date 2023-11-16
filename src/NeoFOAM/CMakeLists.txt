# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM)

include(GNUInstallDirs)

target_include_directories(NeoFOAM PUBLIC
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>")

target_sources(NeoFOAM
  PRIVATE
    "core/Time.cpp"
    "core/Error.cpp"
)
