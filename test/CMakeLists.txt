# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# has to be first since it adds the main target
add_subdirectory(catch2)

function(neofoam_unit_test TEST)
    add_executable(${TEST} "${TEST}.cpp")
    target_link_libraries(${TEST} PRIVATE neofoam_catch_main Kokkos::kokkos NeoFOAM cpptrace::cpptrace)

    add_test(NAME ${TEST} COMMAND mpiexec -n 4 ${CMAKE_BINARY_DIR}/bin/${TEST})
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
endfunction()

add_subdirectory(core)
add_subdirectory(fields)
add_subdirectory(finiteVolume)
add_subdirectory(mesh)
