// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

namespace NeoFOAM
{

enum class cellTypes : std::uint8_t{
    edge,
    quad,
    tri,
    hexah,
    prism,
    tetra,
    pyram,
    total
};

inline const consteval std::size_t totalCellTypes = static_cast<std::size_t>(cellTypes::total);

} // namespace NeoFOAM