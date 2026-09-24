// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/writers.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

void registerFieldWriters(nb::module_& m)
{
    // -------------------------------------------------------------------
    // Field writers — write NeoN fields back through OpenFOAM IO
    // -------------------------------------------------------------------
    m.def(
        "write_scalar_field",
        [](const fvcc::VolumeField<NeoN::scalar>& field, nf::RunTime& rt)
        { nf::write(field, rt.mesh); },
        "field"_a,
        "runtime"_a,
        "Write a NeoN scalar VolumeField via OpenFOAM IO"
    );

    m.def(
        "write_vector_field",
        [](const fvcc::VolumeField<NeoN::Vec3>& field, nf::RunTime& rt)
        { nf::write(field, rt.mesh); },
        "field"_a,
        "runtime"_a,
        "Write a NeoN vector VolumeField via OpenFOAM IO"
    );
}

} // namespace NeoFOAM::bindings
