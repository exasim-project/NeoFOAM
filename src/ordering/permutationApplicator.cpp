// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoFOAM/ordering/permutationApplicator.hpp"

namespace NeoFOAM
{
PermutationApplicator::PermutationApplicator(const NeoN::Executor& exec)
    : exec_ {exec}
{
}
} // namespace NeoFOAM

