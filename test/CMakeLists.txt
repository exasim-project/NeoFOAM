# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# has to be first since it adds the main target
add_subdirectory(catch2)

add_subdirectory(core)
add_subdirectory(fields)
add_subdirectory(finiteVolume)
add_subdirectory(mesh)
