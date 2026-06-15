# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

function(foam_adapter_example NAME)
  add_executable(${NAME} ${NAME}.cpp)
  target_compile_definitions(${NAME} INTERFACE namespaceFoam=1)
  target_link_libraries(${NAME} NeoFOAM)
  install(TARGETS ${NAME})
endfunction()

add_subdirectory(neoIcoFoam)
add_subdirectory(neoPisoFoam)
add_subdirectory(neoPimpleFoam)
add_subdirectory(neoSimpleFoam)
