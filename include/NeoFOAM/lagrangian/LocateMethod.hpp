// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include <string>

namespace NeoFOAM::lagrangian
{


// Wrapper method for locate method, either findCell or walk (fincell is OpenFOAM's method, walk is
// NeoFOAM's method)
enum class LocateMethod
{
    findCell, ///< OpenFOAM mesh.findCell, host
    walk      ///< NeoFOAM walk, executor (GPU-capable) , uses TrackingMesh and CarrierField info
};

/* @brief Map a config string to a LocateMethod enum value. */
inline LocateMethod locateMethodFromName(const std::string& name)
{
    if (name == "findCell") return LocateMethod::findCell;
    if (name == "walk") return LocateMethod::walk;

    NF_ERROR_EXIT("Unknown particle locate method '" + name + "' (valid: findCell walk)");

    return LocateMethod::findCell; // unreachable, silences compiler warning
}

} // namespace NeoFOAM::lagrangian
