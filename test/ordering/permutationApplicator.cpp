// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/ordering/permutationApplicator.hpp"

namespace
{

using NeoFOAM::PermutationApplicator;
}

TEST_CASE("Applying identity permutation leaves mesh unchanged", "[permutationApplicator]")
{
}

TEST_CASE("Applying permutation reorders cell centers")
{}

TEST_CASE("Applying permutation reorders cell volumes")
{}

TEST_CASE("Applying permutation remaps owner indices")
{}

TEST_CASE("Applying permutation remaps neighbour indices")
{}

TEST_CASE("Applying permutation preserves field-cell association")
{}
