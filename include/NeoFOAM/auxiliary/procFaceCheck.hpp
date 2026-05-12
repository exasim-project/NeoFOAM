// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// Cross-rank consistency assertions for proc-face values on Volume / Surface
// fields. Sprinkle these calls into neoIcoFoam.cpp (or any solver) between
// operations to localise where proc-face inconsistency enters during a real
// distributed run. Each call exchanges per-face values with the corresponding
// neighbour rank via MPI_Sendrecv and reports any mismatch.

#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mpi.h>

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{
namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

inline bool valuesMatch(NeoN::scalar a, NeoN::scalar b, double tol)
{
    const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    const double mag = std::max(std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b)));
    return diff <= tol * std::max(1.0, mag);
}

inline bool valuesMatch(NeoN::Vec3 a, NeoN::Vec3 b, double tol)
{
    return valuesMatch(a[0], b[0], tol) && valuesMatch(a[1], b[1], tol)
        && valuesMatch(a[2], b[2], tol);
}

inline std::string formatValue(NeoN::scalar v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6e", static_cast<double>(v));
    return std::string(buf);
}

inline std::string formatValue(NeoN::Vec3 v)
{
    char buf[160];
    std::snprintf(
        buf,
        sizeof(buf),
        "(%.6e, %.6e, %.6e)",
        static_cast<double>(v[0]),
        static_cast<double>(v[1]),
        static_cast<double>(v[2])
    );
    return std::string(buf);
}

inline NeoN::scalar negate(NeoN::scalar v) { return -v; }
inline NeoN::Vec3 negate(NeoN::Vec3 v) { return NeoN::Vec3 {-v[0], -v[1], -v[2]}; }

template<typename ValueType>
struct MpiTraits;

template<>
struct MpiTraits<NeoN::scalar>
{
    static constexpr int countMultiplier = 1;
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

template<>
struct MpiTraits<NeoN::Vec3>
{
    static constexpr int countMultiplier = 3;
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

// Read the proc-tail of boundaryData().value() into a host vector keyed by
// (procPatchIdx, faceWithinPatch).
template<typename Field>
auto extractProcBoundaryValues(const Field& field)
{
    using ValueType = typename Field::VectorValueType;
    auto host = field.boundaryData().value().copyToHost();
    const auto view = host.view();

    const auto& bm = field.mesh().boundaryMesh();
    const auto totalPatches = bm.nBoundaries();
    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto firstProcPatch = totalPatches - procPatchCount;
    const auto& patchOffsets = bm.offset();

    std::vector<std::vector<ValueType>> perPatchValues(procPatchCount);
    for (NeoN::localIdx p = 0; p < procPatchCount; ++p)
    {
        const auto patchIdx = firstProcPatch + p;
        const auto start = patchOffsets[patchIdx];
        const auto end = patchOffsets[patchIdx + 1];
        perPatchValues[p].reserve(end - start);
        for (auto bf = start; bf < end; ++bf)
        {
            perPatchValues[p].push_back(view[bf]);
        }
    }
    return perPatchValues;
}

// For a VolumeField: for each proc face, look up the LOCAL owner cell's
// internal value (this is the "other rank's ghost" from the neighbour's side).
template<typename Field>
auto extractProcOwnerInternalValues(const Field& field)
{
    using ValueType = typename Field::VectorValueType;
    auto hostInternal = field.internalVector().copyToHost();
    const auto internalView = hostInternal.view();

    const auto& bm = field.mesh().boundaryMesh();
    auto faceCellsHost = bm.faceCells().copyToHost();
    const auto faceCellsView = faceCellsHost.view();

    const auto totalPatches = bm.nBoundaries();
    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto firstProcPatch = totalPatches - procPatchCount;
    const auto& patchOffsets = bm.offset();
    const auto nInternalFaces = field.mesh().nInternalFaces();

    std::vector<std::vector<ValueType>> perPatchValues(procPatchCount);
    for (NeoN::localIdx p = 0; p < procPatchCount; ++p)
    {
        const auto patchIdx = firstProcPatch + p;
        const auto start = patchOffsets[patchIdx];
        const auto end = patchOffsets[patchIdx + 1];
        perPatchValues[p].reserve(end - start);
        for (auto bf = start; bf < end; ++bf)
        {
            // bf indexes into boundaryData / faceCells (compressed boundary tail).
            const auto ownerCell = faceCellsView[bf];
            perPatchValues[p].push_back(internalView[ownerCell]);
        }
    }
    return perPatchValues;
}

template<typename Field>
bool fieldIsVolume(const Field& field)
{
    // VolumeField: internalVector size == nCells.
    // SurfaceField: internalVector size == nTotalFaces.
    return field.internalVector().size() == field.mesh().nCells();
}

} // namespace detail

/**
 * @brief Sign convention for proc-face cross-rank comparison.
 *
 * Flux fields (`phi`, `phiHbyA`, anything = `S_f · vec`) have OPPOSITE sign on
 * the two sides of a proc cut because `S_f` points outward from the owner cell
 * — owner differs per rank. That is geometrically correct, not an
 * inconsistency. Pass `SignConvention::FlipExpected` for those fields.
 *
 * Scalar fields like `p`, `rAU` (cell-centred or its surface interpolation),
 * and the boundaryData of a Volume velocity field `U` (= ghost cell value)
 * should match exactly across the cut — use `SignConvention::Equal`.
 */
enum class SignConvention
{
    Equal,
    FlipExpected
};

/**
 * @brief Assert that proc-face boundary values are consistent across ranks.
 *
 * Behaviour by field type:
 *
 *   VolumeField — `boundaryData().value()` at a proc-tail entry holds the
 *     GHOST cell value (= remote rank's owner cell value at the matching
 *     face). The check therefore sends each rank's LOCAL owner-cell value at
 *     the matching face and asserts the receiver's boundaryData == received.
 *     i.e. each rank's view of the neighbour's cell must equal what the
 *     neighbour actually has there. After `correctBoundaryConditions`, this
 *     should hold to floating-point tolerance.
 *
 *   SurfaceField — both ranks store the same physical face value in
 *     `boundaryData().value()`. The check sends each rank's boundaryData and
 *     compares directly. Pass `SignConvention::FlipExpected` for flux fields
 *     (`phi`, `phiHbyA`, anything = `S_f · vec`) which legitimately differ in
 *     sign across the cut.
 *
 * Mismatches print to stderr; returns true if every rank passes (AllReduce).
 *
 * Usage:
 *   nf::checkProcFaceConsistency(p, "p after solve");
 *   nf::checkProcFaceConsistency(U, "U after correctBC");
 *   nf::checkProcFaceConsistency(rAUf, "rAUf");
 *   nf::checkProcFaceConsistency(phi, "phi", 1e-12,
 *                                nf::SignConvention::FlipExpected);
 */
template<typename Field>
bool checkProcFaceConsistency(
    const Field& field,
    const std::string& label,
    double tolerance = 1e-12,
    SignConvention sign = SignConvention::Equal
)
{
    using ValueType = typename Field::VectorValueType;

    const auto& bm = field.mesh().boundaryMesh();
    if (!bm.isDistributed())
    {
        return true;
    }

    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    const auto procPatchCount = bm.nProcBoundaryPatches();
    const auto& nbrRanks = bm.neighbourRank();

    const bool isVolume = detail::fieldIsVolume(field);

    // Local boundaryData (proc-tail) — what we will assert.
    auto perPatchBoundary = detail::extractProcBoundaryValues(field);
    // Local owner-cell internal value at each proc face — what we send for
    // VolumeFields. Only used in the volume case.
    auto perPatchOwner = isVolume
        ? detail::extractProcOwnerInternalValues(field)
        : std::vector<std::vector<ValueType>> {};

    bool allMatch = true;
    for (NeoN::localIdx p = 0; p < procPatchCount; ++p)
    {
        const int nbrRank = static_cast<int>(nbrRanks[p]);
        const int nFaces = static_cast<int>(perPatchBoundary[p].size());

        // Pick what to send: owner-cell internal for VolumeField, boundaryData
        // for SurfaceField.
        const std::vector<ValueType>& sendBuf =
            isVolume ? perPatchOwner[p] : perPatchBoundary[p];

        std::vector<ValueType> remote(nFaces);
        const int countMul = detail::MpiTraits<ValueType>::countMultiplier;

        MPI_Sendrecv(
            sendBuf.data(),
            nFaces * countMul,
            detail::MpiTraits<ValueType>::type(),
            nbrRank,
            7777,
            remote.data(),
            nFaces * countMul,
            detail::MpiTraits<ValueType>::type(),
            nbrRank,
            7777,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        for (int f = 0; f < nFaces; ++f)
        {
            // Sign-flip applies only to surface flux fields. VolumeField
            // ghost values are scalars/vectors with no flip.
            ValueType expected =
                (!isVolume && sign == SignConvention::FlipExpected)
                    ? detail::negate(remote[f])
                    : remote[f];
            // Compare against local boundaryData regardless of field type:
            // that's the value supposed to mirror the neighbour.
            const ValueType& localVal = perPatchBoundary[p][f];
            if (!detail::valuesMatch(localVal, expected, tolerance))
            {
                allMatch = false;
                std::fprintf(
                    stderr,
                    "[procFaceCheck] %s: rank %d patch %d (-> %d) face %d MISMATCH "
                    "local.boundary=%s remote%s=%s\n",
                    label.c_str(),
                    myRank,
                    static_cast<int>(p),
                    nbrRank,
                    f,
                    detail::formatValue(localVal).c_str(),
                    isVolume ? ".internal[ownerOfFace]" : ".boundary",
                    detail::formatValue(remote[f]).c_str()
                );
            }
        }
    }

    // Reduce so every rank knows the global verdict; useful for one-line
    // pass/fail logs.
    int localOk = allMatch ? 1 : 0;
    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (myRank == 0)
    {
        if (globalOk)
        {
            std::fprintf(stderr, "[procFaceCheck] %s: OK\n", label.c_str());
        }
        else
        {
            std::fprintf(stderr, "[procFaceCheck] %s: FAILED (see mismatches above)\n", label.c_str());
        }
    }
    return globalOk != 0;
}

} // namespace NeoFOAM
