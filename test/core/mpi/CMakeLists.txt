# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoN authors

if(NeoN_ENABLE_MPI_SUPPORT)
  neon_unit_test_mpi(fullDuplexCommBuffer MPI_SIZE 3)
  neon_unit_test_mpi(halfDuplexCommBuffer MPI_SIZE 3)
  neon_unit_test_mpi(operators MPI_SIZE 3)
endif()
