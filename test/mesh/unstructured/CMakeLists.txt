# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  neofoam_unit_test(communicator MPI_SIZE 3)
endif()

neofoam_unit_test(unstructuredMesh)
