// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// Full per-step / per-checkpoint state dumps for offline CPU-vs-GPU bisection
// of the multi-rank GPU proc-boundary divergence bug.
//
// This is a *superset* of NeoFOAM/auxiliary/procFaceDump.hpp (which dumps only
// proc-tail values). The proc-face dumps stay gated on `NEOFOAM_PROC_DUMP=1`;
// the new full dumps are gated on **`NEOFOAM_FULL_DUMP=1`** so the cheaper
// proc-face dumps can still be enabled independently for the existing bisect
// scripts (tools/hpc-bisect-gpu-proc-divergence.sh,
// tools/hpc-bisect-option-b.sh).
//
// Output file layout (one tree per run, per-rank subdirectories):
//
//   processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<name>_<kind>.txt
//
// where `<kind>` ∈ {internal, allFaces, A_rowptr, A_colidx, A_values, b, x0, nonLocalA_values, nonLocalA_colIdxs, partition}.
//
// One-shot files written once at step 0 (no step/piso/nonOrth prefix):
//
//   processor{rank}/dumps/geometry_<accessor>.txt        — sf, magSf, cf, faceCells, deltaCoeffs, weights
//   processor{rank}/dumps/proc_meta.txt                  — nCells / row partitioning / neighbour ranks
//
// Row format: ASCII, one entry per line. Floats formatted at `%.10e` so
// FP roundoff (~1e-15) rounds to byte-identical text across CPU/GPU runs
// and any real divergence stands out under strict `diff(1)`.
//
// Standalone Ginkgo CG reproducer (`tools/ginkgo-standalone-cg/`) is designed
// to load these files directly and rebuild the distributed CSR matrix for
// rank-aware CPU↔CUDA replay.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <type_traits>

#include <mpi.h>

#include "NeoN/NeoN.hpp"

namespace NeoFOAM
{

/// True when env var `NEOFOAM_FULL_DUMP` is set to a non-empty, non-"0" value.
/// Cached on first call; lock-free thereafter.
inline bool fullDumpEnabled()
{
    static const bool enabled = []
    {
        const char* v = std::getenv("NEOFOAM_FULL_DUMP");
        return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
}

namespace detail
{

/// Create processor{rank}/dumps/ once per rank.
inline std::filesystem::path ensureDumpDir(int rank)
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::path("processor" + std::to_string(rank)) / "dumps";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

/// Build a per-checkpoint filename:
///   processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<name>_<kind>.txt
inline std::string buildCheckpointFilename(
    int rank,
    int stepIdx,
    int pisoIter,
    int nonOrthIter,
    const std::string& checkpoint,
    const std::string& name,
    const std::string& kind
)
{
    auto dir = ensureDumpDir(rank);
    char fn[512];
    std::snprintf(
        fn,
        sizeof(fn),
        "%s/%04d_%d_%d_%s__%s_%s.txt",
        dir.string().c_str(),
        stepIdx,
        pisoIter,
        nonOrthIter,
        checkpoint.c_str(),
        name.c_str(),
        kind.c_str()
    );
    return std::string(fn);
}

/// Build a one-shot filename:
///   processor{rank}/dumps/<name>.txt
inline std::string buildOneShotFilename(int rank, const std::string& name)
{
    auto dir = ensureDumpDir(rank);
    char fn[512];
    std::snprintf(fn, sizeof(fn), "%s/%s.txt", dir.string().c_str(), name.c_str());
    return std::string(fn);
}

inline void writeRow(std::FILE* f, std::size_t i, NeoN::scalar v)
{
    std::fprintf(f, "%zu %.10e\n", i, static_cast<double>(v));
}

inline void writeRow(std::FILE* f, std::size_t i, NeoN::Vec3 v)
{
    std::fprintf(
        f,
        "%zu %.10e %.10e %.10e\n",
        i,
        static_cast<double>(v[0]),
        static_cast<double>(v[1]),
        static_cast<double>(v[2])
    );
}

template<typename Integral, std::enable_if_t<std::is_integral_v<Integral>, int> = 0>
inline void writeRow(std::FILE* f, std::size_t i, Integral v)
{
    std::fprintf(f, "%zu %lld\n", i, static_cast<long long>(v));
}

template<typename T>
void dumpVectorToFile(const NeoN::Vector<T>& vec, const std::string& filename)
{
    auto host = vec.copyToHost();
    const auto v = host.view();
    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        writeRow(f, i, v[i]);
    }
    std::fclose(f);
}

} // namespace detail


// ---------------------------------------------------------------------------
// Field dumps — internal vector
// ---------------------------------------------------------------------------

/**
 * @brief Dump the FULL internalVector of a Volume/Surface field — one entry
 * per cell (Volume) or per face (Surface). One line per entry.
 *
 * No-op unless `NEOFOAM_FULL_DUMP=1`.
 *
 * Compare this dump rank-by-rank between a CPU-NeoFOAM run and a GPU-NeoFOAM run
 * to localise interior-vs-boundary differences at every PISO checkpoint.
 *
 * Output: processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<name>_internal.txt
 */
template<typename Field>
void dumpInternal(
    const Field& field,
    const std::string& name,
    const std::string& checkpoint,
    int stepIdx,
    int pisoIter,
    int nonOrthIter
)
{
    if (!fullDumpEnabled()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string filename =
        detail::buildCheckpointFilename(rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "internal");
    detail::dumpVectorToFile(field.internalVector(), filename);
}


// ---------------------------------------------------------------------------
// SurfaceField dumps — the *full* face range
// ---------------------------------------------------------------------------

/**
 * @brief Dump the FULL face data of a SurfaceField across the full face range
 * (internal faces + non-proc boundary + proc tail) — one line per face.
 *
 * Two segments are written into the same file separated by a comment:
 *   - lines `0 .. nTotalFaces-1`        : internalVector (compressed face indexing)
 *   - lines `nTotalFaces..end`          : boundaryData().value() (boundary patch tail)
 *
 * Both are needed because of the SurfaceField dual-storage invariant
 * (internalVector[proc-tail] == boundaryData[proc-tail] LOCAL pre-exchange,
 * != post-exchange where boundaryData holds the neighbour value).
 *
 * Compare this dump CPU vs GPU at each PISO checkpoint to see which segment
 * diverges first — internal-face math, non-proc BC, or proc-tail exchange.
 *
 * No-op unless `NEOFOAM_FULL_DUMP=1`. No-op for VolumeField (we look at the
 * internalVector().size() == nCells convention to detect).
 *
 * Output: processor{rank}/dumps/<step:04d>_<piso>_<nonOrth>_<checkpoint>__<name>_allFaces.txt
 */
template<typename SurfaceField>
void dumpAllFaces(
    const SurfaceField& field,
    const std::string& name,
    const std::string& checkpoint,
    int stepIdx,
    int pisoIter,
    int nonOrthIter
)
{
    if (!fullDumpEnabled()) return;

    // SurfaceField has internalVector size == nTotalFaces; reject VolumeField.
    if (field.internalVector().size() == field.mesh().nCells())
    {
        // Volume field passed by mistake — caller should use dumpInternal.
        return;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string filename =
        detail::buildCheckpointFilename(rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "allFaces");

    auto internalHost = field.internalVector().copyToHost();
    auto boundaryHost = field.boundaryData().value().copyToHost();
    const auto iv = internalHost.view();
    const auto bv = boundaryHost.view();

    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "# section internalVector  size=%zu\n", iv.size());
    for (std::size_t i = 0; i < iv.size(); ++i)
    {
        detail::writeRow(f, i, iv[i]);
    }
    std::fprintf(f, "# section boundaryData    size=%zu\n", bv.size());
    for (std::size_t i = 0; i < bv.size(); ++i)
    {
        detail::writeRow(f, i, bv[i]);
    }
    std::fclose(f);
}


// ---------------------------------------------------------------------------
// LinearSystem dump in distributed form, with row partitioning
// ---------------------------------------------------------------------------

/**
 * @brief Dump the *rank-local block* of a distributed LinearSystem in CSR form,
 * the rhs and the initial iterate, AND the row partitioning needed to rebuild
 * a distributed matrix in `tools/ginkgo-standalone-cg/`.
 *
 * Files produced (per rank, per call site):
 *   <prefix>__<name>_A_rowptr.txt   — local CSR row offsets (size nLocalRows + 1)
 *   <prefix>__<name>_A_colidx.txt   — local CSR column indices (size nnz_local)
 *   <prefix>__<name>_A_values.txt   — local CSR values        (size nnz_local)
 *   <prefix>__<name>_nonLocalA_values.txt  — non-local COO values
 *   <prefix>__<name>_nonLocalA_colIdxs.txt — non-local COO column indices (global ghost IDs)
 *   <prefix>__<name>_b.txt          — rhs                     (size nLocalRows)
 *   <prefix>__<name>_x0.txt         — initial iterate          (size nLocalRows)
 *   <prefix>__<name>_partition.txt  — single line "<rank> <nProcs> <nLocalRows> <globalRowStart>"
 *
 * The row partitioning is range-partition style and is constructed via
 * MPI_Exscan over `nLocalRows` so the standalone Ginkgo test can rebuild a
 * distributed `gko::experimental::distributed::Matrix` from these files alone.
 *
 * No-op unless `NEOFOAM_FULL_DUMP=1`. Call AFTER `PDESolver::assemble()` /
 * `PDESolver::assemble(rhs)` so the LinearSystem is fully populated.
 *
 * For VolumeField<scalar> systems the values are scalar; for Vec3-valued
 * systems they are Vec3 (one row per Vec3 component-wise, dumped as a vector).
 */
template<typename LinearSystem, typename InitialIterate>
void dumpDistLinearSystem(
    const LinearSystem& ls,
    const InitialIterate& x0,
    const std::string& name,
    const std::string& checkpoint,
    int stepIdx,
    int pisoIter,
    int nonOrthIter
)
{
    if (!fullDumpEnabled()) return;

    int rank = 0;
    int nProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // Local block: rowOffs.size() - 1 == number of locally-owned rows.
    auto rowOffsHost = ls.matrix().sparsity()->rowOffs().copyToHost();
    const auto rowOffs = rowOffsHost.view();
    const long long nLocalRows = static_cast<long long>(
        rowOffs.size() == 0 ? 0 : rowOffs.size() - 1
    );

    // Range-partition start row via MPI_Exscan over local row counts.
    long long globalRowStart = 0;
    MPI_Exscan(&nLocalRows, &globalRowStart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    // rank 0's Exscan output is undefined; force it to 0
    if (rank == 0) globalRowStart = 0;

    // CSR local block
    detail::dumpVectorToFile(
        ls.matrix().sparsity()->rowOffs(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "A_rowptr"
        )
    );
    detail::dumpVectorToFile(
        ls.matrix().sparsity()->colIdxs(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "A_colidx"
        )
    );
    detail::dumpVectorToFile(
        ls.matrix().values(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "A_values"
        )
    );

    // Non-local block (one COO entry per proc-boundary face per row)
    detail::dumpVectorToFile(
        ls.nonLocalMatrix().values(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "nonLocalA_values"
        )
    );
    detail::dumpVectorToFile(
        ls.nonLocalMatrix().sparsity()->colIdxs(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "nonLocalA_colIdxs"
        )
    );
    detail::dumpVectorToFile(
        ls.nonLocalMatrix().sparsity()->rowOffs(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "nonLocalA_rowOffs"
        )
    );

    // rhs and x0
    detail::dumpVectorToFile(
        ls.rhs(),
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "b"
        )
    );
    detail::dumpVectorToFile(
        x0,
        detail::buildCheckpointFilename(
            rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "x0"
        )
    );

    // Per-call partition info (single-line file).
    const std::string partFile =
        detail::buildCheckpointFilename(rank, stepIdx, pisoIter, nonOrthIter, checkpoint, name, "partition");
    std::FILE* pf = std::fopen(partFile.c_str(), "w");
    if (pf)
    {
        std::fprintf(
            pf,
            "%d %d %lld %lld\n",
            rank,
            nProcs,
            nLocalRows,
            globalRowStart
        );
        std::fclose(pf);
    }
}


// ---------------------------------------------------------------------------
// Geometry one-shot dump — step 0 only
// ---------------------------------------------------------------------------

/**
 * @brief Dump per-rank mesh geometry — written once. Idempotent (re-writes are
 * harmless and produce identical output).
 *
 * Files (under processor{rank}/dumps/):
 *   geometry_sf.txt          — internal face area vectors            (size nInternalFaces)
 *   geometry_cf.txt          — internal face centres                 (size nInternalFaces)
 *   geometry_faceCells.txt   — boundary faceCells map (compressed)   (size nBoundaryFaces + nProcFaces)
 *   geometry_bm_sf.txt       — boundary face area vectors            (compressed)
 *   geometry_bm_magSf.txt    — boundary face magnitudes              (compressed)
 *   geometry_bm_cf.txt       — boundary face centres                 (compressed)
 *   geometry_bm_deltaCoeffs.txt — boundary face delta coefficients   (compressed)
 *   geometry_bm_weights.txt  — boundary face interpolation weights   (compressed)
 *
 * Useful when comparing CPU and GPU runs: identical geometry is a precondition
 * for matching the assembled LinearSystem coefficients.
 *
 * No-op unless `NEOFOAM_FULL_DUMP=1`.
 */
inline void dumpGeometry(const NeoN::UnstructuredMesh& mesh)
{
    if (!fullDumpEnabled()) return;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Internal faces — note these are OF-full-sized (see project memory:
    // mesh.faceAreas() / faceCentres() include empty patches). For the
    // *internal* segment they're correct; the proc-face entries live in
    // bm.sf() / bm.magSf() (compressed) which we dump separately below.
    detail::dumpVectorToFile(mesh.faceAreas(), detail::buildOneShotFilename(rank, "geometry_sf"));
    detail::dumpVectorToFile(mesh.faceCentres(), detail::buildOneShotFilename(rank, "geometry_cf"));

    const auto& bm = mesh.boundaryMesh();
    detail::dumpVectorToFile(bm.faceCells(),    detail::buildOneShotFilename(rank, "geometry_faceCells"));
    detail::dumpVectorToFile(bm.sf(),           detail::buildOneShotFilename(rank, "geometry_bm_sf"));
    detail::dumpVectorToFile(bm.magSf(),        detail::buildOneShotFilename(rank, "geometry_bm_magSf"));
    detail::dumpVectorToFile(bm.cf(),           detail::buildOneShotFilename(rank, "geometry_bm_cf"));
    detail::dumpVectorToFile(bm.deltaCoeffs(),  detail::buildOneShotFilename(rank, "geometry_bm_deltaCoeffs"));
    detail::dumpVectorToFile(bm.weights(),      detail::buildOneShotFilename(rank, "geometry_bm_weights"));
}


// ---------------------------------------------------------------------------
// Per-rank metadata one-shot — step 0 only
// ---------------------------------------------------------------------------

/**
 * @brief Dump per-rank meta: nCells, range-partition row offset, MPI size,
 * count and list of proc-patch neighbour ranks.
 *
 * Single-line + tabular content (one entry per line after header), one file
 * per rank:
 *
 *   processor{rank}/dumps/proc_meta.txt
 *
 * Contents:
 *
 *   rank <r>
 *   nProcs <p>
 *   nCells <c>
 *   nInternalFaces <if>
 *   nBoundaryFaces <bf>
 *   nProcBoundaryPatches <pp>
 *   nProcBoundaryFaces <pf>
 *   globalRowStart <g>      # computed via MPI_Exscan over nCells (range partition)
 *   globalNCells <total>
 *   neighbourRanks <r0> <r1> ...
 *
 * Consumed by `tools/ginkgo-standalone-cg/` to rebuild the distributed
 * row-partition for a Ginkgo replay of the dumped LinearSystem.
 *
 * No-op unless `NEOFOAM_FULL_DUMP=1`.
 */
inline void dumpProcMeta(const NeoN::UnstructuredMesh& mesh)
{
    if (!fullDumpEnabled()) return;

    int rank = 0;
    int nProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    const long long nCells = static_cast<long long>(mesh.nCells());

    long long globalRowStart = 0;
    MPI_Exscan(&nCells, &globalRowStart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) globalRowStart = 0;

    long long globalNCells = 0;
    MPI_Allreduce(&nCells, &globalNCells, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const auto& bm = mesh.boundaryMesh();

    // neighbourRank() is a host-side std::vector<localIdx>, not a NeoN::Vector
    const auto& nbView = bm.neighbourRank();

    const std::string fn = detail::buildOneShotFilename(rank, "proc_meta");
    std::FILE* f = std::fopen(fn.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "rank %d\n", rank);
    std::fprintf(f, "nProcs %d\n", nProcs);
    std::fprintf(f, "nCells %lld\n", nCells);
    std::fprintf(f, "nInternalFaces %lld\n", static_cast<long long>(mesh.nInternalFaces()));
    std::fprintf(f, "nBoundaryFaces %lld\n", static_cast<long long>(bm.nBoundaryFaces()));
    std::fprintf(
        f,
        "nProcBoundaryPatches %lld\n",
        static_cast<long long>(bm.nProcBoundaryPatches())
    );
    std::fprintf(
        f,
        "nProcBoundaryFaces %lld\n",
        static_cast<long long>(bm.nProcBoundaryFaces())
    );
    std::fprintf(f, "globalRowStart %lld\n", globalRowStart);
    std::fprintf(f, "globalNCells %lld\n", globalNCells);
    std::fprintf(f, "neighbourRanks");
    for (std::size_t i = 0; i < nbView.size(); ++i)
    {
        std::fprintf(f, " %lld", static_cast<long long>(nbView[i]));
    }
    std::fprintf(f, "\n");
    std::fclose(f);
}

} // namespace NeoFOAM
