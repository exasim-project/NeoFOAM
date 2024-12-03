# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  neofoam_unit_test_mpi(fullDuplexCommBuffer MPI_SIZE 3)
  neofoam_unit_test_mpi(halfDuplexCommBuffer MPI_SIZE 3)
  neofoam_unit_test_mpi(operators MPI_SIZE 3)
endif()
