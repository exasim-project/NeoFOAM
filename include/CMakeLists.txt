# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_library(NeoFOAM_public_api INTERFACE) # dummy target

if(${NEOFOAM_DP_SCALAR})
  target_compile_definitions(NeoFOAM_public_api PUBLIC NEOFOAM_DP_SCALAR=1)
endif()
if(${NEOFOAM_DP_LABEL})
  target_compile_definitions(NeoFOAM_public_api PUBLIC NEOFOAM_DP_LABEL=1)
endif()

set_property(
  TARGET NeoFOAM_public_api
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(TARGET NeoFOAM_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                       "${CMAKE_CURRENT_BINARY_DIR}")
