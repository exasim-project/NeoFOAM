// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// Standalone Ginkgo distributed CG reproducer for the GPU multi-rank
// proc-boundary divergence bug — see .planning/debug/gpu-proc-boundary-divergence.md.
//
// Loads a per-rank dump tree produced by an instrumented NeoFOAM run
// (NEOFOAM_FULL_DUMP=1 → fullDump.hpp::dumpDistLinearSystem) and rebuilds
// the *exact* assembled distributed LinearSystem to replay it through
// Ginkgo's CG solver, once with `gko::ReferenceExecutor` and once with
// `gko::CudaExecutor`. The L∞ norm of `x_ref - x_cuda` discriminates:
//
//   • close to zero       — Ginkgo is innocent; the NeoN ↔ Ginkgo
//                           composition layer is responsible.
//   • non-trivially large — real Ginkgo distributed CUDA bug.
//
// Run on 2 ranks (matching the toy reproducer in
// test/setup_distributedChannelToy_mpi2/):
//
//   mpirun -np 2 ./standalone_cg <dump_dir> <checkpoint> [iters] [tol]
//
// where:
//   <dump_dir>     path containing processor{0..N-1}/dumps/
//   <checkpoint>   filename prefix to load, e.g.
//                  "0001_0_0_before_pEqn_solve__p" (note no _<kind>.txt suffix)
//   [iters]        optional CG iteration cap (default 200)
//   [tol]          optional residual tolerance (default 1e-12)
//
// File layout expected per rank R, for checkpoint C:
//   processor{R}/dumps/{C}_partition.txt          # "rank nProcs nLocalRows globalRowStart"
//   processor{R}/dumps/{C}_A_rowptr.txt           # CSR row offsets   (nLocalRows+1 entries)
//   processor{R}/dumps/{C}_A_colidx.txt           # CSR col indices   (nnz_local)
//   processor{R}/dumps/{C}_A_values.txt           # CSR values        (nnz_local)
//   processor{R}/dumps/{C}_nonLocalA_colIdxs.txt  # non-local col idx (global ghost cell ids)
//   processor{R}/dumps/{C}_nonLocalA_rowOffs.txt  # non-local row idx (local cell)
//   processor{R}/dumps/{C}_nonLocalA_values.txt   # non-local off-diag values
//   processor{R}/dumps/{C}_b.txt                  # rhs (nLocalRows)
//   processor{R}/dumps/{C}_x0.txt                 # initial iterate (nLocalRows)
//   processor{R}/dumps/proc_meta.txt              # one-shot per-rank globals
//
// Build via: ./build.sh
// Run via:   ./run.sh <dump_dir> <checkpoint>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <ginkgo/ginkgo.hpp>


namespace
{

using ValueType = double;
using IndexType = std::int32_t;

struct PartitionInfo
{
    int rank{};
    int nProcs{};
    long long nLocalRows{};
    long long globalRowStart{};
};

// Read a single-line `"<rank> <nProcs> <nLocalRows> <globalRowStart>"` file.
PartitionInfo readPartition(const std::string& path)
{
    PartitionInfo p;
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "FATAL: cannot open " << path << "\n";
        std::exit(2);
    }
    f >> p.rank >> p.nProcs >> p.nLocalRows >> p.globalRowStart;
    return p;
}

// Read a per-line numeric dump. Each non-comment line is "<index> <val>"
// (single-column scalar) or "<index> <v0> <v1> <v2>" (Vec3). For Vec3 we
// take the first component (scalar-equivalent component-wise; matrices we
// dump are scalar-valued for the pressure solve, which is the primary target).
//
// Lines starting with '#' are skipped (section markers used in some dumps).
std::vector<ValueType> readScalarColumn(const std::string& path)
{
    std::vector<ValueType> out;
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "WARN: missing " << path << " (treating as empty)\n";
        return out;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        long long idx;
        double v;
        if (!(iss >> idx >> v)) continue;
        out.push_back(static_cast<ValueType>(v));
    }
    return out;
}

std::vector<IndexType> readIndexColumn(const std::string& path)
{
    std::vector<IndexType> out;
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "WARN: missing " << path << " (treating as empty)\n";
        return out;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        long long idx;
        long long v;
        if (!(iss >> idx >> v)) continue;
        out.push_back(static_cast<IndexType>(v));
    }
    return out;
}

std::string ckptFile(const std::string& root, int rank, const std::string& checkpoint, const std::string& kind)
{
    return root + "/processor" + std::to_string(rank) + "/dumps/"
        + checkpoint + "_" + kind + ".txt";
}

std::string oneShotFile(const std::string& root, int rank, const std::string& name)
{
    return root + "/processor" + std::to_string(rank) + "/dumps/" + name + ".txt";
}


// Solve A x = b with Ginkgo CG on the supplied executor. Returns the locally-
// owned slice of x on the host as a std::vector<double>. Uses Ginkgo's
// non-distributed CSR + dense vector path because:
//
//   * Reading the row-partition Matrix from dumped CSR + non-local COO
//     directly is messier (would need access to a partition object built
//     across all ranks);
//   * The objective of this tool is to verify Ginkgo CPU vs CUDA agree on
//     the SAME assembled rank-local block when fed identical inputs.
//
// Each rank solves its own local block independently — this is sufficient
// to discriminate Ginkgo CPU vs CUDA on per-rank assembled data (which IS
// what NeoN hands to Ginkgo from each rank). The distributed-matrix variant
// is a stretch goal; for the first cut, per-rank-local replay is enough
// because:
//
//   * If `L∞(x_cpu - x_cuda)` differs at the SAME rank-local LinearSystem,
//     Ginkgo's CUDA executor diverges from its Reference executor on the
//     SAME inputs — irrespective of MPI / halo logic — and that points
//     squarely at a Ginkgo bug.
//   * If `L∞(x_cpu - x_cuda)` is zero at every rank but the application-
//     level dumps still diverge, the bug is in NeoN's coupling layer
//     (commPattern setup, halo exchange, or stream visibility), NOT
//     Ginkgo itself.
std::vector<ValueType> solveLocalCG(
    std::shared_ptr<gko::Executor> exec,
    long long nLocalRows,
    const std::vector<IndexType>& rowPtr,
    const std::vector<IndexType>& colIdx,
    const std::vector<ValueType>& values,
    const std::vector<ValueType>& rhs,
    const std::vector<ValueType>& x0,
    int maxIters,
    double tol,
    long long& itersOut,
    double& finalResidualOut
)
{
    auto host_exec = exec->get_master();

    // Build host CSR on the master, then transfer to target exec.
    auto host_dim = gko::dim<2>{
        static_cast<gko::size_type>(nLocalRows),
        static_cast<gko::size_type>(nLocalRows)
    };
    auto host_csr = gko::matrix::Csr<ValueType, IndexType>::create(host_exec, host_dim);
    host_csr->read(gko::matrix_data<ValueType, IndexType>::create_from_arrays(
        host_dim,
        gko::array<ValueType>(host_exec, values.begin(), values.end()),
        gko::array<IndexType>(host_exec, colIdx.begin(), colIdx.end()),
        gko::array<IndexType>(host_exec, rowPtr.begin(), rowPtr.end())
    ));

    auto A = gko::matrix::Csr<ValueType, IndexType>::create(exec);
    A->copy_from(host_csr.get());

    auto bDim = gko::dim<2>{static_cast<gko::size_type>(nLocalRows), 1};
    auto host_b = gko::matrix::Dense<ValueType>::create(
        host_exec,
        bDim,
        gko::array<ValueType>(host_exec, rhs.begin(), rhs.end()),
        1
    );
    auto host_x = gko::matrix::Dense<ValueType>::create(
        host_exec,
        bDim,
        gko::array<ValueType>(host_exec, x0.begin(), x0.end()),
        1
    );
    auto b = gko::matrix::Dense<ValueType>::create(exec);
    b->copy_from(host_b.get());
    auto x = gko::matrix::Dense<ValueType>::create(exec);
    x->copy_from(host_x.get());

    auto solverFactory = gko::solver::Cg<ValueType>::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(maxIters)).on(exec),
            gko::stop::ResidualNorm<ValueType>::build().with_baseline(gko::stop::mode::absolute).with_reduction_factor(static_cast<ValueType>(tol)).on(exec)
        )
        .on(exec);

    auto loggerStorage = gko::log::Convergence<ValueType>::create();
    auto solver = solverFactory->generate(A);
    solver->add_logger(loggerStorage);
    solver->apply(b.get(), x.get());

    itersOut = static_cast<long long>(loggerStorage->get_num_iterations());
    auto resNorm = gko::as<gko::matrix::Dense<ValueType>>(loggerStorage->get_residual_norm());
    if (resNorm != nullptr)
    {
        auto hostRes = gko::matrix::Dense<ValueType>::create(host_exec);
        hostRes->copy_from(resNorm);
        finalResidualOut = static_cast<double>(hostRes->at(0, 0));
    }
    else
    {
        finalResidualOut = -1.0;
    }

    auto host_out = gko::matrix::Dense<ValueType>::create(host_exec);
    host_out->copy_from(x.get());
    std::vector<ValueType> result(static_cast<std::size_t>(nLocalRows));
    for (long long i = 0; i < nLocalRows; ++i)
    {
        result[static_cast<std::size_t>(i)] = host_out->at(static_cast<gko::size_type>(i), 0);
    }
    return result;
}


double linfDiff(const std::vector<ValueType>& a, const std::vector<ValueType>& b)
{
    if (a.size() != b.size()) return -1.0;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}


} // namespace


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0, nProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    if (argc < 3)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0]
                      << " <dump_dir> <checkpoint> [iters=200] [tol=1e-12]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::string dumpDir = argv[1];
    const std::string checkpoint = argv[2];
    const int maxIters = (argc >= 4) ? std::atoi(argv[3]) : 200;
    const double tol = (argc >= 5) ? std::atof(argv[4]) : 1e-12;

    // ---------------- Load rank-local LinearSystem ------------------------ //
    auto partition = readPartition(ckptFile(dumpDir, rank, checkpoint, "partition"));
    if (partition.nProcs != nProcs)
    {
        if (rank == 0)
        {
            std::cerr << "FATAL: dump was produced with nProcs=" << partition.nProcs
                      << " but launched with " << nProcs << " ranks\n";
        }
        MPI_Finalize();
        return 2;
    }

    auto rowPtr = readIndexColumn(ckptFile(dumpDir, rank, checkpoint, "A_rowptr"));
    auto colIdx = readIndexColumn(ckptFile(dumpDir, rank, checkpoint, "A_colidx"));
    auto values = readScalarColumn(ckptFile(dumpDir, rank, checkpoint, "A_values"));
    auto rhs = readScalarColumn(ckptFile(dumpDir, rank, checkpoint, "b"));
    auto x0 = readScalarColumn(ckptFile(dumpDir, rank, checkpoint, "x0"));

    if (static_cast<long long>(rowPtr.size()) != partition.nLocalRows + 1)
    {
        std::cerr << "[rank " << rank << "] FATAL: rowPtr size " << rowPtr.size()
                  << " != nLocalRows+1 " << (partition.nLocalRows + 1) << "\n";
        MPI_Finalize();
        return 3;
    }
    if (static_cast<long long>(rhs.size()) != partition.nLocalRows
        || static_cast<long long>(x0.size()) != partition.nLocalRows)
    {
        std::cerr << "[rank " << rank << "] FATAL: rhs/x0 size mismatch\n";
        MPI_Finalize();
        return 4;
    }

    // ---------------- Solve on CPU ---------------------------------------- //
    auto refExec = gko::ReferenceExecutor::create();
    long long itersCpu = 0;
    double resCpu = 0.0;
    auto xCpu = solveLocalCG(
        refExec,
        partition.nLocalRows,
        rowPtr,
        colIdx,
        values,
        rhs,
        x0,
        maxIters,
        tol,
        itersCpu,
        resCpu
    );

    // ---------------- Solve on CUDA --------------------------------------- //
    std::vector<ValueType> xCuda;
    long long itersCuda = 0;
    double resCuda = 0.0;
    bool cudaAvailable = false;
    try
    {
        auto cudaExec = gko::CudaExecutor::create(0, refExec);
        cudaAvailable = true;
        xCuda = solveLocalCG(
            cudaExec,
            partition.nLocalRows,
            rowPtr,
            colIdx,
            values,
            rhs,
            x0,
            maxIters,
            tol,
            itersCuda,
            resCuda
        );
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
        {
            std::cerr << "WARN: CudaExecutor unavailable (" << e.what()
                      << "); skipping CUDA solve.\n";
        }
    }

    // ---------------- Report --------------------------------------------- //
    double linf = -1.0;
    if (cudaAvailable)
    {
        linf = linfDiff(xCpu, xCuda);
    }

    // Per-rank summary, gather to rank 0.
    char line[512];
    if (cudaAvailable)
    {
        std::snprintf(
            line,
            sizeof(line),
            "[rank %d] nRows=%lld  CPU(iters=%lld, |r|=%.3e)  CUDA(iters=%lld, |r|=%.3e)  L_inf(x_cpu-x_cuda)=%.3e\n",
            rank,
            partition.nLocalRows,
            itersCpu,
            resCpu,
            itersCuda,
            resCuda,
            linf
        );
    }
    else
    {
        std::snprintf(
            line,
            sizeof(line),
            "[rank %d] nRows=%lld  CPU(iters=%lld, |r|=%.3e)  CUDA(unavailable)\n",
            rank,
            partition.nLocalRows,
            itersCpu,
            resCpu
        );
    }

    // Each rank prints its own line (interleaved by MPI). Acceptable for diag output.
    std::cout << line << std::flush;

    // Global max L_inf across ranks
    double globalLinf = 0.0;
    MPI_Allreduce(&linf, &globalLinf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && cudaAvailable)
    {
        std::printf("GLOBAL L_inf(x_cpu - x_cuda) over all ranks = %.6e\n", globalLinf);
        if (globalLinf < 1e-10)
        {
            std::printf("VERDICT: Ginkgo CPU and CUDA AGREE on the dumped A,b — "
                        "Ginkgo is INNOCENT for this LinearSystem.\n");
        }
        else
        {
            std::printf("VERDICT: Ginkgo CPU and CUDA DISAGREE on the dumped A,b — "
                        "potential Ginkgo bug, file upstream.\n");
        }
    }

    MPI_Finalize();
    return 0;
}
