# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_subdirectory(mpi)

neofoam_unit_test(dictionary)
neofoam_unit_test(executor)
neofoam_unit_test(registerClass)
neofoam_unit_test(parallelAlgorithms)
