// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors


#include "NeoFOAM/primitives/vector.hpp"
#include "NeoFOAM/primitives/scalar.hpp"
#include <Kokkos_Core.hpp>

namespace NeoFOAM
{

std::ostream& operator<<(std::ostream& os, const Vector& vec) {
    os << "(" << vec[0] << " " << vec[1] << " " << vec[2] << ")";
    return os;
}


} // namespace NeoFOAM