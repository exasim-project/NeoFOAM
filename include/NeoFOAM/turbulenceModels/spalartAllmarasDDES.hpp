// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/pdeSolver.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;
using scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;

namespace NeoFOAM
{

/**
 * @brief NeoFOAM-level wrapper for the Spalart-Allmaras DDES turbulence model.
 *
 * Mirrors the OpenFOAM turbulence->validate() / turbulence->correct() interface.
 * Owns all intermediate fields and cached operators so the solver only needs to
 * manage the primary fields (nuTilda, nut) and call validate() once at setup and
 * correct() after each momentum solve.
 *
 * Two wall-distance fields are required (they are distinct):
 *  - wallDist     : cell-centred distances from Foam::wallDist::y(), used in DDES shielding
 *  - nearWallDist : boundary-face distances from turbulenceModel::y()[patchi], used by
 *                   the nutUSpaldingWallFunction boundary condition
 */
class SpalartAllmarasDDES
{
public:

    struct Coefficients
    {
        scalar sigmaNut = 0.66666;
        scalar kappa = 0.41;
        scalar Cb1 = 0.1355;
        scalar Cb2 = 0.622;
        scalar Cw2 = 0.3;
        scalar Cw3 = 2.0;
        scalar Cv1 = 7.1;
        scalar Ct3 = 1.2;
        scalar Ct4 = 0.5;
        scalar Cs = 0.3;
        scalar Cdes = 0.65;
        scalar fdCoef = 8.0;
        scalar fwStar = 0.424;
    };

    /**
     * @brief Construct the turbulence model wrapper.
     *
     * @param exec       Kokkos executor (Serial/CPU/GPU)
     * @param mesh       NeoN unstructured mesh
     * @param nu         Laminar kinematic viscosity (cell-centred)
     * @param wallDist   Cell-centred wall distances (Foam::wallDist::y())
     * @param nearWallDist Boundary-face wall distances (turbulenceModel::y()[patchi])
     * @param delta      LES filter width (from LESModel::delta())
     */
    SpalartAllmarasDDES(
        const NeoN::Executor& exec,
        const NeoN::UnstructuredMesh& mesh,
        const nnfvcc::VolumeField<scalar>& nu,
        const nnfvcc::VolumeField<scalar>& wallDist,
        const nnfvcc::VolumeField<scalar>& nearWallDist,
        const nnfvcc::VolumeField<scalar>& delta
    );

    /**
     * @brief Initialise nut from the current nuTilda. Call once at setup.
     *
     * Equivalent to OpenFOAM's turbulence->validate().
     *
     * @param U       Initial velocity field (needed for wall-function BCs on nut)
     * @param nuTilda Initial SA transported variable
     * @param nut     Turbulent viscosity (updated in place)
     */
    void validate(
        const nnfvcc::VolumeField<Vec3>& U,
        nnfvcc::VolumeField<scalar>& nuTilda,
        nnfvcc::VolumeField<scalar>& nut
    );

    /**
     * @brief Full turbulence update: compute sources, solve nuTilda PDE, update nut.
     *
     * Equivalent to OpenFOAM's turbulence->correct().
     * Call after each momentum/PISO iteration.
     *
     * @param U       Current velocity field
     * @param phi     Face flux field
     * @param nuTilda Transported SA variable (updated in place)
     * @param nut     Turbulent viscosity (updated in place)
     * @param rt      NeoFOAM RunTime (provides time step, scheme dicts)
     */
    void correct(
        const nnfvcc::VolumeField<Vec3>& U,
        nnfvcc::SurfaceField<scalar>& phi,
        nnfvcc::VolumeField<scalar>& nuTilda,
        nnfvcc::VolumeField<scalar>& nut,
        RunTime& rt
    );

    /// @brief Effective viscosity on faces: nu + nut (for momentum equation laplacian)
    nnfvcc::SurfaceField<scalar>& nuEff();

    /// @brief Effective nuTilda diffusion coefficient on faces: (nu + nuTilda)/sigmaNut
    nnfvcc::SurfaceField<scalar>& nuTildaEff();

    /// @brief Velocity gradient tensor (updated each correct() call, for viscousStress term)
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const;

    /**
     * @brief Returns the deviatoric stress at all boundary faces as Vector<SymmTensor>.
     *
     * Computes -nuEff * symm(twoSymm(gradU)).dev2() at each boundary face.
     * Slice per-patch using mesh_.boundaryMesh().patchRange(patchi).
     * Used by forces functionObject to integrate viscous forces over patches.
     */
    NeoN::Vector<NeoN::SymmTensor> devRhoReff() const;

    /// @brief Read-only access to model coefficients (for testing/inspection)
    const Coefficients& coeffs() const { return coeffs_; }

    /// @brief Computed cw1 coefficient (for testing/inspection)
    scalar cw1() const { return cw1_; }

    // Physics kernels — public to allow unit-testing of individual steps
    void correctNut(
        nnfvcc::VolumeField<scalar>& nutField,
        nnfvcc::SurfaceField<scalar>& nutF,
        nnfvcc::SurfaceField<scalar>& nuEffF,
        const nnfvcc::VolumeField<scalar>& nuTilde,
        const nnfvcc::VolumeField<scalar>& nu,
        const nnfvcc::SurfaceField<scalar>& nuF,
        const nnfvcc::VolumeField<Vec3>& u,
        const nnfvcc::VolumeField<scalar>& nearWallDist
    ) const;

    void calcNuTildaDiffusionCoeff(
        nnfvcc::VolumeField<scalar>& nuTilde,
        const nnfvcc::SurfaceField<scalar>& nuF,
        nnfvcc::SurfaceField<scalar>& surfNuTilde,
        nnfvcc::SurfaceField<scalar>& nuTildeEffF
    ) const;

    void
    calcMagSqrVec(nnfvcc::VolumeField<scalar>& magSqr, const nnfvcc::VolumeField<Vec3>& in) const;

    void computeProdSpDDES(
        nnfvcc::VolumeField<scalar>& productionField,
        nnfvcc::VolumeField<scalar>& spCoeffField,
        const nnfvcc::VolumeField<scalar>& nuTildeField,
        const nnfvcc::VolumeField<scalar>& nuField,
        const nnfvcc::VolumeField<NeoN::Tensor>& gradUField,
        const nnfvcc::VolumeField<scalar>& wallDistanceField,
        const nnfvcc::VolumeField<scalar>& deltaField,
        const nnfvcc::VolumeField<scalar>& gradNuTildeMagSqrField
    ) const;

private:

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;

    // Constant physics inputs (held by reference — must outlive this object)
    const nnfvcc::VolumeField<scalar>& nu_;
    const nnfvcc::VolumeField<scalar>& wallDist_;
    const nnfvcc::VolumeField<scalar>& nearWallDist_;
    const nnfvcc::VolumeField<scalar>& delta_;

    // Cached face-interpolated nu (computed once in constructor)
    nnfvcc::SurfaceField<scalar> surfNu_;

    // Owned intermediate fields
    nnfvcc::VolumeField<NeoN::Tensor> gradU_;
    nnfvcc::VolumeField<Vec3> gradNuTilda_;
    nnfvcc::VolumeField<scalar> magSqrGradNuTilda_;
    nnfvcc::VolumeField<scalar> production_;
    nnfvcc::VolumeField<scalar> spCoeff_;
    nnfvcc::SurfaceField<scalar> surfNut_;
    nnfvcc::SurfaceField<scalar> surfNuTilda_;
    nnfvcc::SurfaceField<scalar> nuEff_;
    nnfvcc::SurfaceField<scalar> nuTildaEff_;

    // Cached operators (constructed once)
    nnfvcc::GaussGreenGrad gradOp_;
    nnfvcc::SurfaceInterpolation<scalar> surfInterp_;

    // SA-DDES model coefficients
    Coefficients coeffs_;
    scalar cw1_;
};

} // namespace NeoFOAM
