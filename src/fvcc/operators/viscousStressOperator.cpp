// SPDX-FileCopyrightText: 2024 - 2026 NeoFOAM authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "NeoFOAM/fvcc/operators/viscousStressOperator.hpp"

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

namespace
{

KOKKOS_INLINE_FUNCTION
void atomicAddVec3(Vec3* dst, const Vec3& v)
{
    Kokkos::atomic_add(&(*dst)[0], v[0]);
    Kokkos::atomic_add(&(*dst)[1], v[1]);
    Kokkos::atomic_add(&(*dst)[2], v[2]);
}

KOKKOS_INLINE_FUNCTION
void atomicSubVec3(Vec3* dst, const Vec3& v)
{
    Kokkos::atomic_sub(&(*dst)[0], v[0]);
    Kokkos::atomic_sub(&(*dst)[1], v[1]);
    Kokkos::atomic_sub(&(*dst)[2], v[2]);
}

} // namespace

// ----------------------------
// computeDevStress
// ----------------------------

VolumeField<Tensor> computeDevStress(
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU
)
{
    const UnstructuredMesh& mesh = gradU.mesh();
    const auto exec = gradU.exec();

    VolumeField<Tensor>
        tau(exec, "devStress", mesh, createCalculatedBCs<VolumeBoundary<Tensor>>(mesh));

    // tau = nuEff * dev2(gradU^T)
    // dev2(M)_ij = M_ij - (2/3)*tr(M)*delta_ij
    // (gradU^T)_ij = gradU_ji  =>  tr(gradU^T) = tr(gradU)

    // Internal cells
    {
        const auto [gradUV, nuV, nutV, tauV] = views(
            gradU.internalVector(),
            nu.internalVector(),
            nut.internalVector(),
            tau.internalVector()
        );

        parallelFor(
            exec,
            {0, static_cast<localIdx>(gradU.internalVector().size())},
            NEON_LAMBDA(const localIdx i) {
                const scalar nuEff = nuV[i] + nutV[i];
                const Tensor& g = gradUV[i];
                const scalar diag = (scalar(2) / scalar(3)) * g.trace();
                // dev2(gradU^T) = gradU^T - (2/3)*tr(gradU)*I
                tauV[i] = Tensor(
                              g(0, 0) - diag,
                              g(1, 0),
                              g(2, 0),
                              g(0, 1),
                              g(1, 1) - diag,
                              g(2, 1),
                              g(0, 2),
                              g(1, 2),
                              g(2, 2) - diag
                          )
                        * nuEff;
            },
            "computeDevStress::internal"
        );
    }

    // Boundary faces
    {
        const auto [gradUB, nuB, nutB, tauB] = views(
            gradU.boundaryData().value(),
            nu.boundaryData().value(),
            nut.boundaryData().value(),
            tau.boundaryData().value()
        );

        parallelFor(
            exec,
            {0, static_cast<localIdx>(gradU.boundaryData().value().size())},
            NEON_LAMBDA(const localIdx bf) {
                const scalar nuEff = nuB[bf] + nutB[bf];
                const Tensor& g = gradUB[bf];
                const scalar diag = (scalar(2) / scalar(3)) * g.trace();
                // dev2(gradU^T) = gradU^T - (2/3)*tr(gradU)*I
                tauB[bf] = Tensor(
                               g(0, 0) - diag,
                               g(1, 0),
                               g(2, 0),
                               g(0, 1),
                               g(1, 1) - diag,
                               g(2, 1),
                               g(0, 2),
                               g(1, 2),
                               g(2, 2) - diag
                           )
                         * nuEff;
            },
            "computeDevStress::boundary"
        );
    }

    return tau;
}

// ----------------------------
// divDevReff
// ----------------------------

void divDevReff(
    const SurfaceInterpolation<Tensor>& surfInterp,
    const VolumeField<Tensor>& tau,
    Vector<Vec3>& rhs,
    dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = tau.mesh();
    const auto exec = tau.exec();

    // Interpolate stress tensor to faces
    auto calcSurfBC = createCalculatedBCs<SurfaceBoundary<Tensor>>(mesh);
    SurfaceField<Tensor> tauF(exec, "tauF", mesh, calcSurfBC);
    surfInterp.interpolate(tau, tauF);

    const auto [owner, neighbour, faceCells] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

    const auto [Sf, tauFV, vol, rhsV] =
        views(mesh.faceNormals(), tauF.internalVector(), mesh.cellVolumes(), rhs);

    const localIdx nIF = mesh.nInternalFaces();
    const localIdx nFaces = tauF.size();

    parallelFor(
        exec,
        {0, nIF},
        NEON_LAMBDA(const localIdx f) {
            const Vec3 flux = scalar(-1.0) * (tauFV[f] & Sf[f]);
            atomicAddVec3(&rhsV[owner[f]], flux);
            atomicSubVec3(&rhsV[neighbour[f]], flux);
        },
        "divDevReff::internal"
    );

    parallelFor(
        exec,
        {nIF, nFaces},
        NEON_LAMBDA(const localIdx f) {
            const Vec3 flux = scalar(-1.0) * (tauFV[f] & Sf[f]);
            atomicAddVec3(&rhsV[faceCells[f - nIF]], flux);
        },
        "divDevReff::boundary"
    );

    parallelFor(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) { rhsV[c] *= operatorScaling[c] / vol[c]; },
        "divDevReff::normalize"
    );
}

// ----------------------------
// GaussViscousStress entry points
// ----------------------------

void GaussViscousStress::explicitOp(
    Vector<Vec3>& rhs,
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU,
    const dsl::Coeff operatorScaling
) const
{
    auto tau = computeDevStress(nu, nut, gradU);
    divDevReff(surfaceInterpolationTensor_, tau, rhs, operatorScaling);
}

void ViscousStressOperator::implicitOperation(la::LinearSystem<Vec3>& ls) const
{
    NF_ASSERT(viscousOp_, "ViscousStressOperatorStrategy not initialized");
    NF_ASSERT(gradUTensor_, "gradU not initialized");
    const auto& mesh = nu_.mesh();
    auto& rhs = ls.rhs();

    Vector<Vec3> contribution(rhs.exec(), rhs.size(), zero<Vec3>());
    viscousOp_->explicitOp(contribution, nu_, nut_, *gradUTensor_, this->getCoefficient());

    const auto [vol, contrib, rhsV] = views(mesh.cellVolumes(), contribution, rhs);
    parallelFor(
        rhs.exec(),
        {0, static_cast<localIdx>(rhs.size())},
        NEON_LAMBDA(const localIdx i) { rhsV[i] -= contrib[i] * vol[i]; }
    );
}

VolumeField<Vec3> GaussViscousStress::viscousStress(
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU,
    const dsl::Coeff operatorScaling
) const
{
    VolumeField<Vec3> result(
        exec_,
        "div((nuEff*dev2(T(grad(U)))))",
        mesh_,
        createCalculatedBCs<VolumeBoundary<Vec3>>(mesh_)
    );
    fill(result.internalVector(), zero<Vec3>());
    fill(result.boundaryData().value(), zero<Vec3>());
    auto tau = computeDevStress(nu, nut, gradU);
    divDevReff(surfaceInterpolationTensor_, tau, result.internalVector(), operatorScaling);
    return result;
}

void GaussViscousStress::viscousStress(
    VolumeField<Vec3>& result,
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU,
    const dsl::Coeff operatorScaling
) const
{
    fill(result.internalVector(), zero<Vec3>());
    fill(result.boundaryData().value(), zero<Vec3>());
    auto tau = computeDevStress(nu, nut, gradU);
    divDevReff(surfaceInterpolationTensor_, tau, result.internalVector(), operatorScaling);
}

void GaussViscousStress::viscousStress(
    Vector<Vec3>& result,
    const VolumeField<scalar>& nu,
    const VolumeField<scalar>& nut,
    const VolumeField<Tensor>& gradU,
    const dsl::Coeff operatorScaling
) const
{
    auto tau = computeDevStress(nu, nut, gradU);
    divDevReff(surfaceInterpolationTensor_, tau, result, operatorScaling);
}

} // namespace NeoN::finiteVolume::cellCentred
