// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/core/primitives/vector.hpp"

namespace NeoFOAM
{

std::ostream& operator<<(std::ostream& os, const Vector& vec)
{
    os << "(" << vec[0] << " " << vec[1] << " " << vec[2] << ")";
    return os;
}

} // namespace NeoFOAM
