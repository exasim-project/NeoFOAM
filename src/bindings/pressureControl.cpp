// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/auxiliary/pressureControl.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM::bindings
{

void registerPressureControl(nb::module_& m)
{
    nb::class_<NeoFOAM::PressureControl>(m, "PressureControl")
        .def(
            "__init__",
            [](NeoFOAM::PressureControl* self,
               const fvcc::VolumeField<NeoN::scalar>& p,
               const fvcc::VolumeField<NeoN::scalar>& rho,
               std::optional<NeoN::scalar> pMin,
               std::optional<NeoN::scalar> pMax,
               std::optional<NeoN::scalar> pMinFactor,
               std::optional<NeoN::scalar> pMaxFactor,
               std::optional<NeoN::scalar> rhoMin,
               std::optional<NeoN::scalar> rhoMax)
            {
                new (self) NeoFOAM::PressureControl(
                    p, rho, {pMin, pMax, pMinFactor, pMaxFactor, rhoMin, rhoMax}
                );
            },
            "p"_a,
            "rho"_a,
            "p_min"_a = nb::none(),
            "p_max"_a = nb::none(),
            "p_min_factor"_a = nb::none(),
            "p_max_factor"_a = nb::none(),
            "rho_min"_a = nb::none(),
            "rho_max"_a = nb::none(),
            "OpenFOAM's pressureControl for the compressible solvers (pRefRequired = false): "
            "the limits come from absolute p_min/p_max, or from the factors / rho limits applied "
            "to the value-fixing p patches"
        )
        .def_prop_ro("active", &NeoFOAM::PressureControl::active)
        .def_prop_ro("p_min", &NeoFOAM::PressureControl::pMin)
        .def_prop_ro("p_max", &NeoFOAM::PressureControl::pMax)
        .def(
            "limit",
            &NeoFOAM::PressureControl::limit,
            "p"_a,
            "Clip p to [p_min, p_max]; True if limits are configured (the caller then "
            "re-evaluates p's boundary conditions), like pressureControl::limit()"
        );
}

} // namespace NeoFOAM::bindings
