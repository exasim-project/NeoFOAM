// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/auxiliary/pressureControl.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace NeoFOAM
{

namespace fvcc = NeoN::finiteVolume::cellCentred;
using NeoN::localIdx;
using NeoN::scalar;

namespace
{

struct MinMax
{
    scalar min = std::numeric_limits<scalar>::max();
    scalar max = std::numeric_limits<scalar>::lowest();
};

// Free functions: NEON_LAMBDA is an extended __device__ lambda under CUDA, which nvcc does not
// allow in a constructor or inside another lambda.
MinMax minMaxOf(const NeoN::Executor& exec, const NeoN::Vector<scalar>& vec, localIdx start, localIdx end)
{
    MinMax r;
    if (end <= start) return r;
    const auto v = vec.view();
    Kokkos::Min<scalar> minReducer(r.min);
    NeoN::parallelReduce(
        exec,
        {start, end},
        NEON_LAMBDA(const localIdx i, scalar& m) { m = Kokkos::min(m, v[i]); },
        minReducer
    );
    Kokkos::Max<scalar> maxReducer(r.max);
    NeoN::parallelReduce(
        exec,
        {start, end},
        NEON_LAMBDA(const localIdx i, scalar& m) { m = Kokkos::max(m, v[i]); },
        maxReducer
    );
    return r;
}

MinMax combine(const MinMax& a, const MinMax& b)
{
    return {std::min(a.min, b.min), std::max(a.max, b.max)};
}

// The extremes over all ranks; a serial run keeps the local ones.
MinMax reduceAcrossRanks(const NeoN::UnstructuredMesh& mesh, MinMax r)
{
#ifdef NF_WITH_MPI_SUPPORT
    if (mesh.boundaryMesh().isDistributed())
    {
        NeoN::mpi::Environment env;
        MPI_Allreduce(MPI_IN_PLACE, &r.min, 1, NeoN::mpi::getType<scalar>(), MPI_MIN, env.comm());
        MPI_Allreduce(MPI_IN_PLACE, &r.max, 1, NeoN::mpi::getType<scalar>(), MPI_MAX, env.comm());
    }
#else
    (void)mesh;
#endif
    return r;
}

void clipInternal(
    const NeoN::Executor& exec,
    NeoN::Vector<scalar>& vec,
    bool clipMin,
    scalar pMin,
    bool clipMax,
    scalar pMax
)
{
    auto v = vec.view();
    NeoN::parallelFor(
        exec,
        {0, static_cast<localIdx>(vec.size())},
        NEON_LAMBDA(const localIdx i) {
            if (clipMax) v[i] = Kokkos::min(v[i], pMax);
            if (clipMin) v[i] = Kokkos::max(v[i], pMin);
        },
        "pressureControl::clip"
    );
}

// OpenFOAM's pressureControl looks for patches whose fvPatchField::fixesValue() is true, and that
// includes every mixed patch (mixedFvPatchField::fixesValue() is true). NeoN keeps fixesValue = false
// for mixed patches, because it drives the gradient treatment, and flags them with `mixed` instead.
// The requires-expression keeps this building against a NeoN that predates the attribute.
template<class Attributes>
bool fixesValueLikeOpenFOAM(const Attributes& attributes)
{
    if constexpr (requires { attributes.mixed; })
    {
        return attributes.fixesValue || attributes.mixed;
    }
    else
    {
        return attributes.fixesValue;
    }
}

[[noreturn]] void missingReference(const char* wanted, const char* instead)
{
    throw std::runtime_error(
        std::string("pressureControl: '") + wanted + "' specified rather than '" + instead
        + "' but the corresponding reference pressure cannot be evaluated from the boundary "
          "conditions: no p patch fixes its value. Please specify '" + instead + "'."
    );
}

} // namespace

PressureControl::PressureControl(
    const fvcc::VolumeField<scalar>& p,
    const fvcc::VolumeField<scalar>& rho,
    const PressureLimitSettings& s
)
{
    if (s.pMax && s.pMin)
    {
        pMax_ = *s.pMax;
        pMin_ = *s.pMin;
        limitMaxP_ = limitMinP_ = true;
    }
    else
    {
        // Extremes of p and rho over the patches that fix p's value.
        MinMax pRef;
        MinMax rhoRef;
        bool pLimits = false;
        const auto bcs = p.boundaryConditions();
        for (localIdx patchID = 0; patchID < static_cast<localIdx>(bcs.size()); ++patchID)
        {
            if (!fixesValueLikeOpenFOAM(bcs[static_cast<std::size_t>(patchID)].attributes())) continue;
            pLimits = true;
            const auto [start, end] = p.boundaryData().range(patchID);
            pRef = combine(pRef, minMaxOf(p.exec(), p.boundaryData().value(), start, end));
            rhoRef = combine(rhoRef, minMaxOf(rho.exec(), rho.boundaryData().value(), start, end));
        }
        pRef = reduceAcrossRanks(p.mesh(), pRef);
        rhoRef = reduceAcrossRanks(p.mesh(), rhoRef);

        if (s.pMax)
        {
            pMax_ = *s.pMax;
            limitMaxP_ = true;
        }
        else if (s.pMaxFactor)
        {
            if (!pLimits) missingReference("pMaxFactor", "pMax");
            pMax_ = pRef.max * *s.pMaxFactor;
            limitMaxP_ = true;
        }
        else if (s.rhoMax)
        {
            if (!pLimits) missingReference("rhoMax", "pMax");
            pMax_ = std::max(*s.rhoMax / rhoRef.max, scalar(1)) * pRef.max;
            limitMaxP_ = true;
        }

        if (s.pMin)
        {
            pMin_ = *s.pMin;
            limitMinP_ = true;
        }
        else if (s.pMinFactor)
        {
            if (!pLimits) missingReference("pMinFactor", "pMin");
            pMin_ = pRef.min * *s.pMinFactor;
            limitMinP_ = true;
        }
        else if (s.rhoMin)
        {
            if (!pLimits) missingReference("rhoMin", "pMin");
            pMin_ = std::min(*s.rhoMin / rhoRef.min, scalar(1)) * pRef.min;
            limitMinP_ = true;
        }
    }

    if (active())
    {
        NeoN::Logging::info(
            "pressureControl: pMin {} pMax {}",
            limitMinP_ ? std::to_string(pMin_) : std::string("(none)"),
            limitMaxP_ ? std::to_string(pMax_) : std::string("(none)")
        );
    }
}

bool PressureControl::limit(fvcc::VolumeField<scalar>& p) const
{
    if (!active()) return false;

    const auto exec = p.exec();
    const auto& pb = p.boundaryData().value();
    MinMax r = combine(
        minMaxOf(exec, p.internalVector(), 0, static_cast<localIdx>(p.internalVector().size())),
        minMaxOf(exec, pb, 0, static_cast<localIdx>(pb.size()))
    );
    r = reduceAcrossRanks(p.mesh(), r);

    const bool clipMax = limitMaxP_ && r.max > pMax_;
    const bool clipMin = limitMinP_ && r.min < pMin_;
    if (clipMax) NeoN::Logging::info("pressureControl: p max {}", r.max);
    if (clipMin) NeoN::Logging::info("pressureControl: p min {}", r.min);
    if (clipMax || clipMin)
    {
        clipInternal(exec, p.internalVector(), clipMin, pMin_, clipMax, pMax_);
        NeoN::fence(exec);
    }
    return true;
}

} // namespace NeoFOAM
