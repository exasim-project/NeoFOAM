# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

find_package(pybind11 CONFIG)

pybind11_add_module(pyNeoFOAM ${PROJECT_SOURCE_DIR}/bindings/python.cpp)

target_include_directories(pyNeoFOAM PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(pyNeoFOAM PRIVATE NeoFOAM Kokkos::kokkos
                                        pybind11::pybind11)
