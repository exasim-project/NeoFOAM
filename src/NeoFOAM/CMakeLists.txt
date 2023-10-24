# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM)

target_sources(NeoFOAM
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
    "hello_world.cxx"
)
