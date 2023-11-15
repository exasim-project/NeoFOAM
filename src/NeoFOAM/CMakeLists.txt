# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM)
add_library(NeoFOAM::NeoFOAM ALIAS NeoFOAM)

target_include_directories(NeoFOAM PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_sources(NeoFOAM
  PUBLIC
    "core/Time.cpp"
    "core/Error.cpp"
)
