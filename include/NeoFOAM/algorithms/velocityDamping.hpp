// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors
#pragma once

#include "NeoN/NeoN.hpp"

#include <type_traits>

namespace NeoFOAM
{

/**
 * @brief Port of OpenFOAM's fv::velocityDampingConstraint (fvOptions).
 *
 * NeoFOAM has no fvOptions layer, so a case that ships
 *
 *     velocityDamping { type velocityDampingConstraint; selectionMode all; UMax 300; }
 *
 * in system/fvOptions gets that constraint applied by simpleFoam and silently
 * NOT applied by neoSimpleFoam. On the occDrivAer cold start this is not
 * cosmetic: the reference run damps 12 cells on iteration 1 (and none of the
 * remaining 3999), i.e. it is exactly the startup safety net that keeps the
 * initial transient bounded.
 *
 * Faithful to upstream `addDamping` (velocityDampingConstraint.C), which is a
 * DIAGONAL-ONLY constraint -- there is no source/rhs contribution:
 *
 *     scale = cbrt(V_c)^2
 *     diag[c] += C * scale * (|U_c| - UMax)      for cells with |U_c| > UMax
 *
 * The comment upstream explains the scaling: the added term is deltaU/deltaT
 * with a local time scale deltaT = cbrt(V)/U, and since the diagonal is
 * manipulated directly it is multiplied by the cell volume, leaving V^(2/3).
 * Boosting the diagonal without a matching source drags the offending cells'
 * velocity back toward zero; it is a constraint, not a conservative term, and
 * it vanishes from the converged solution once no cell exceeds UMax.
 *
 * Applied to the ALREADY RELAXED system, matching simpleFoam's ordering
 * (`UEqn.relax(); fvOptions.constrain(UEqn);`) -- see PDE::setPostRelaxConstraint.
 *
 * Only `selectionMode all` is supported; the caller is responsible for
 * rejecting cell-set restricted variants.
 *
 * @param ls    The assembled (and relaxed) linear system; its diagonal is modified in place.
 * @param U     The velocity field whose current internal values decide which cells are damped.
 * @param UMax  Velocity magnitude above which damping engages.
 * @param C     Upstream's `C` coefficient (default 1, as in fv::velocityDampingConstraint).
 * @return Number of cells damped on THIS rank (caller reduces across ranks for reporting).
 */
template<typename LinearSystemType, typename VolumeFieldType>
NeoN::label applyVelocityDamping(
    LinearSystemType& ls,
    const VolumeFieldType& U,
    NeoN::scalar UMax,
    NeoN::scalar C = 1.0
)
{
    // The segregated momentum system (PDE<Vec3>) has scalar coefficients; a coupled Vec3
    // matrix would need the boost on every diagonal component.
    static_assert(
        std::is_same_v<
            std::remove_cv_t<std::remove_reference_t<decltype(ls.matrix().values().view()[0])>>,
            NeoN::scalar>,
        "applyVelocityDamping expects a scalar-coefficient (segregated) linear system"
    );

    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    // Same addressing the relaxation kernel uses: diagIdx(celli) maps a cell to
    // its diagonal slot in the CSR values array.
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto uV = U.internalVector().view();
    const auto volV = U.mesh().cellVolumes().view();

    const auto nCells = static_cast<NeoN::localIdx>(uV.size());

    NeoN::label nDamped = 0;
    NeoN::parallelReduce(
        U.exec(),
        {0, nCells},
        NEON_LAMBDA(const NeoN::localIdx celli, NeoN::label& count) {
            const NeoN::scalar magU = NeoN::mag(uV[celli]);
            if (magU > UMax)
            {
                // scale = sqr(cbrt(V)) -- V^(2/3), not V.
                const NeoN::scalar cbrtV = Kokkos::cbrt(volV[celli]);
                matrix.values[ma.diagIdx(celli)] += C * cbrtV * cbrtV * (magU - UMax);
                ++count;
            }
        },
        nDamped
    );

    return nDamped;
}

} // namespace NeoFOAM
