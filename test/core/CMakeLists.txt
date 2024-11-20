# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  add_subdirectory(mpi)
endif()

add_subdirectory(primitives)

neofoam_unit_test(dictionary)
neofoam_unit_test(tokenList)
neofoam_unit_test(inputs)
neofoam_unit_test(executor)
neofoam_unit_test(parallelAlgorithms)
neofoam_unit_test(registerClass)
