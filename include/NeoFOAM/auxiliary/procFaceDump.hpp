// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// Per-rank proc-boundary value dumps for offline CPU-vs-GPU bisection.
//
// Gated entirely on env NEOFOAM_PROC_DUMP=1 — when unset, every call is a
// no-op (single getenv at first call, cached). For the bisect workflow see
// tools/hpc-bisect-gpu-proc-divergence.sh.
//
// Output one file per (step, pisoIter, nonOrthIter, checkpoint, fieldName):
//   processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<field>.txt
//
// Each row is "<patch> <face> <value...>" with values formatted at %.10e so
// CPU↔GPU FP roundoff (~1e-15) rounds to identical text and real divergence
// (the bug we're hunting) is visible to strict diff(1).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include <mpi.h>

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/auxiliary/procFaceCheck.hpp"

namespace NeoFOAM
{

inline bool procDumpEnabled()
{
    static const bool enabled = std::getenv("NEOFOAM_PROC_DUMP") != nullptr;
    return enabled;
}

namespace detail
{

inline void dumpRow(std::FILE* f, int patch, int face, NeoN::scalar v)
{
    std::fprintf(f, "%d %d %.10e\n", patch, face, static_cast<double>(v));
}

inline void dumpRow(std::FILE* f, int patch, int face, NeoN::Vec3 v)
{
    std::fprintf(
        f,
        "%d %d %.10e %.10e %.10e\n",
        patch,
        face,
        static_cast<double>(v[0]),
        static_cast<double>(v[1]),
        static_cast<double>(v[2])
    );
}

} // namespace detail

/**
 * @brief Dump proc-tail boundaryData().value() for one field to a per-rank file.
 *
 * No-op unless NEOFOAM_PROC_DUMP=1 is set in the environment, and no-op for
 * non-distributed meshes. Reuses detail::extractProcBoundaryValues from
 * procFaceCheck.hpp (host-side proc-tail walk).
 *
 * Filename: processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<fieldName>.txt
 * Row format: "<patchIdx> <faceWithinPatch> <value0> [<value1> <value2>]"
 *
 * Pair every call with the existing nf::checkProcFaceConsistency at the same
 * checkpoint so the dump file traces the exact value sequence the consistency
 * check sees.
 */
template<typename Field>
void dumpProcFaces(
    const Field& field,
    const std::string& fieldName,
    const std::string& checkpoint,
    int stepIdx,
    int pisoIter,
    int nonOrthIter
)
{
    if (!procDumpEnabled()) return;

    const auto& bm = field.mesh().boundaryMesh();
    if (!bm.isDistributed()) return;

    const auto procPatchCount = bm.nProcBoundaryPatches();
    if (procPatchCount == 0) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto perPatchBoundary = detail::extractProcBoundaryValues(field);

    namespace fs = std::filesystem;
    const fs::path dir = fs::path("processor" + std::to_string(rank)) / "dumps";
    std::error_code ec;
    fs::create_directories(dir, ec);

    char filename[512];
    std::snprintf(
        filename,
        sizeof(filename),
        "%s/%04d_%d_%d_%s__%s.txt",
        dir.string().c_str(),
        stepIdx,
        pisoIter,
        nonOrthIter,
        checkpoint.c_str(),
        fieldName.c_str()
    );

    std::FILE* f = std::fopen(filename, "w");
    if (!f) return;

    for (NeoN::localIdx p = 0; p < procPatchCount; ++p)
    {
        const auto nFaces = perPatchBoundary[p].size();
        for (std::size_t bf = 0; bf < nFaces; ++bf)
        {
            detail::dumpRow(
                f,
                static_cast<int>(p),
                static_cast<int>(bf),
                perPatchBoundary[p][bf]
            );
        }
    }
    std::fclose(f);
}

} // namespace NeoFOAM
