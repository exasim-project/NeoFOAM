// SPDX-FileCopyrightText: 2024-2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <nanobind/nanobind.h>

namespace NeoFOAM::bindings
{

void registerRuntime(nanobind::module_& m);
void registerFieldFactories(nanobind::module_& m);
void registerFieldWriters(nanobind::module_& m);
void registerPDESolver(nanobind::module_& m);
void registerPressureVelocityCoupling(nanobind::module_& m);
void registerUtility(nanobind::module_& m);
void registerExplicitOperators(nanobind::module_& m);
void registerTurbulenceModel(nanobind::module_& m);
void registerWallFunctions(nanobind::module_& m);
void registerFieldRelaxation(nanobind::module_& m);

} // namespace NeoFOAM::bindings
