// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

/*
A Permutation is an immutable value type representing a bijective mapping between
two index spaces and providing efficient operations on that mapping.

Invariant:

- oldToNew_.size() == newToOld_.size()

- oldToNew_ is bijective

- newToOld_ is the inverse of oldToNew_

- indices lie in [0,size())
*/
#pragma once

#include "NeoN/NeoN.hpp"

#include <cstddef>
#include <span>
#include <vector>

class Permutation
{
public:

    using IndexType = NeoN::label;

    explicit Permutation(std::vector<IndexType> oldToNew);

    [[nodiscard]]
    static Permutation identity(std::size_t size);

    [[nodiscard]]
    std::size_t size() const noexcept
    {
        return oldToNew_.size();
    }

    [[nodiscard]]
    IndexType oldToNew(IndexType oldIndex) const;

    [[nodiscard]]
    IndexType newToOld(IndexType newIndex) const;

    [[nodiscard]]
    std::span<const IndexType> oldToNew() const noexcept;

    [[nodiscard]]
    std::span<const IndexType> newToOld() const noexcept;

    [[nodiscard]]
    Permutation inverse() const;

    // [[nodiscard]]
    // Permutation compose(
    //     const Permutation& other
    // ) const;

    [[nodiscard]]
    bool isIdentity() const noexcept;

private:

    std::vector<IndexType> oldToNew_;
    std::vector<IndexType> newToOld_;
};