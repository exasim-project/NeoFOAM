// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"
#include "volFields.H"
#include <string>
#include "NeoN/mesh/unstructured/pointInterpolation.hpp"



namespace NeoFOAM::lagrangian
{
// wrapper method for interpolation, either cell or cellPoint (OpenFOAM's volPointInterpolation)
enum class Interpolation
{
    cell,
    cellPoint
};

/* @brief Map a config string to an Interpolation enum value. */
inline Interpolation interpolationFromName(const std::string& name)
{
    if (name == "cell") return Interpolation::cell;
    if (name == "cellPoint") return Interpolation::cellPoint;

    NF_ERROR_EXIT("Unknown carrier interpolation '" + name + "' (valid: cell cellPoint)");

    return Interpolation::cell; // unreachable, silences compiler warning
}

// Struct to hold carrier field, cell and point values, copied to the executor
struct CarrierField
{
    Interpolation mode;
    NeoN::Vector<NeoN::Vec3> U; ///< cell values
    NeoN::Vector<NeoN::Vec3>
        Upoint; ///< point values (cellPoint only), OpenFOAM volPointInterpolation
};

// Struct to hold read-only views of CarrierField, captured by value inside kernels
struct CarrierFieldView
{
    bool cellPoint;
    NeoN::View<const NeoN::Vec3> U;
    NeoN::View<const NeoN::Vec3> Upoint;
};

// Build a read-only view of a CarrierField, captured by value inside kernels.
inline CarrierFieldView view(const CarrierField& f)
{
    return {f.mode == Interpolation::cellPoint, f.U.view(), f.Upoint.view()};
}

// Declare a function to build a CarrierField from an OpenFOAM volVectorField and an Interpolation
// mode, on a specified executor.
CarrierField
buildCarrierField(const NeoN::Executor& exec, const Foam::volVectorField& U, Interpolation mode);


// Q2: carrier velocity at x in cell c. cell: the cell value. cellPoint: NeoN's
// interpolateCellPoint (OpenFOAM's interpolationCellPoint) from the cell and point values.
KOKKOS_INLINE_FUNCTION
NeoN::Vec3 interpolate(
    const CarrierFieldView& f,
    const NeoN::TrackingMeshView& m,
    const NeoN::label c,
    const NeoN::Vec3& x
)
{
    if (!f.cellPoint)
    {
        return f.U[c];
    }
    return NeoN::interpolateCellPoint<NeoN::Vec3>(m, f.U, f.Upoint, c, x);
}

} // namespace NeoFOAM::lagrangian
