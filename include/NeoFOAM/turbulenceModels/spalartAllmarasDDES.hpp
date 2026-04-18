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

    // Low-level NeoN physics model
    NeoN::turbulenceModels::SpalartAllmarasDDES physicsModel_;
};

} // namespace NeoFOAM
