# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_subdirectory(core)
add_subdirectory(fields)
add_subdirectory(mesh)
add_subdirectory(finiteVolume)
