// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>

#include "bindings.hpp"

namespace nb = nanobind;

// ===========================================================================
// Module definition
// ===========================================================================
NB_MODULE(neofoam_bindings, m)
{
    m.doc() = "NeoFOAM Python bindings - building blocks for NeoN-based solvers";

    NeoFOAM::bindings::registerRuntime(m);
    NeoFOAM::bindings::registerFieldFactories(m);
    NeoFOAM::bindings::registerFieldWriters(m);
    NeoFOAM::bindings::registerPDESolver(m);
    NeoFOAM::bindings::registerPressureVelocityCoupling(m);
    NeoFOAM::bindings::registerUtility(m);
    NeoFOAM::bindings::registerExplicitOperators(m);
    NeoFOAM::bindings::registerTurbulenceModel(m);
    NeoFOAM::bindings::registerWallFunctions(m);
    NeoFOAM::bindings::registerFieldRelaxation(m);
}
