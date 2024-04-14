# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

pybind11_add_module(pyNeoFOAM ${PROJECT_SOURCE_DIR}/bindings/python.cpp)

target_include_directories(pyNeoFOAM PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(pyNeoFOAM PRIVATE NeoFOAM pybind11::pybind11)
