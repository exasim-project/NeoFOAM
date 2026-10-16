#include "NeoFOAM/ordering/permutation.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace NeoFOAM
{

Permutation::Permutation(std::vector<IndexType> oldToNew)
    : oldToNew_ {std::move(oldToNew)}
    , newToOld_(oldToNew_.size())
{
    const auto permutationSize = oldToNew_.size();

    // Every entry in newToOld_ is initially marked as unassigned.
    std::vector<bool> newIndexAssigned(permutationSize, false);

    for (NeoN::localIdx oldIndex = 0; oldIndex < NeoN::toLocalIdx(permutationSize); ++oldIndex)
    {
        const auto newIndex = oldToNew_[oldIndex];

        // Check negative values before converting newIndex to std::size_t.
        if constexpr (std::is_signed_v<IndexType>)
        {
            if (newIndex < 0)
            {
                throw std::invalid_argument {"Permutation contains a negative new index"};
            }
        }

        const auto newIndexAsSize = static_cast<std::size_t>(newIndex);

        if (newIndexAsSize >= permutationSize)
        {
            throw std::invalid_argument {"Permutation contains a new index outside the valid range"
            };
        }

        if (newIndexAssigned[newIndexAsSize])
        {
            throw std::invalid_argument {"Permutation contains a duplicate new index"};
        }

        newIndexAssigned[newIndexAsSize] = true;

        newToOld_[newIndexAsSize] = static_cast<IndexType>(oldIndex);
    }
}


Permutation Permutation::identity(const NeoN::sizeType size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<IndexType>::max()))
    {
        throw std::length_error {"Permutation size exceeds the range representable by IndexType"};
    }

    std::vector<IndexType> oldToNew(size);

    std::iota(oldToNew.begin(), oldToNew.end(), IndexType {0});

    return Permutation {std::move(oldToNew)};
}


Permutation::IndexType Permutation::oldToNew(const IndexType oldIndex) const
{
    if constexpr (std::is_signed_v<IndexType>)
    {
        if (oldIndex < 0)
        {
            throw std::out_of_range {"Old index is outside the permutation range"};
        }
    }

    const auto index = static_cast<std::size_t>(oldIndex);

    if (index >= size())
    {
        throw std::out_of_range {"Old index is outside the permutation range"};
    }

    return oldToNew_[index];
}


Permutation::IndexType Permutation::newToOld(const IndexType newIndex) const
{
    if constexpr (std::is_signed_v<IndexType>)
    {
        if (newIndex < 0)
        {
            throw std::out_of_range {"New index is outside the permutation range"};
        }
    }

    const auto index = static_cast<std::size_t>(newIndex);

    if (index >= size())
    {
        throw std::out_of_range {"New index is outside the permutation range"};
    }

    return newToOld_[index];
}


std::span<const Permutation::IndexType> Permutation::oldToNew() const noexcept { return oldToNew_; }


std::span<const Permutation::IndexType> Permutation::newToOld() const noexcept { return newToOld_; }


Permutation Permutation::inverse() const { return Permutation {newToOld_}; }


bool Permutation::isIdentity() const noexcept
{
    const NeoN::localIdx size = NeoN::toLocalIdx(oldToNew_.size());

    for (NeoN::localIdx index = 0; index < size; ++index)
    {
        if (oldToNew_[index] != static_cast<IndexType>(index))
        {
            return false;
        }
    }

    return true;
}

} // namespace NeoFOAM
