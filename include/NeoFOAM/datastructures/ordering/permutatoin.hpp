// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*
A Permutation is an immutable value type representing a bijective mapping between 
two index spaces and providing efficient operations on that mapping.
*/
#pragma once
#include "NeoN/NeoN.hpp"

class Permutation
{
    public:
        using IndexType = NeoN::label;

        explicit Permutation(std::vector<IndexType> oldToNew);

        static Permutation identity(std::size_t size);

        [[nodiscard]] std::size_t size() const { return oldToNew_.size(); }

        [[nodiscard]] IndexType oldToNew(IndexType oldIndex) const;

        [[nodiscard]] IndexType newToOld(IndexType newIndex) const;

        [[nodiscard]] Permutation inverse() const;

        [[nodiscard]] Permutation compose(const Permutation& other) const;
}