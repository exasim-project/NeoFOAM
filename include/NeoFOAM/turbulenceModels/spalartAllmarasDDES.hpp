// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/pde.hpp"
#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;
using scalar = NeoN::scalar;
using Vec3 = NeoN::Vec3;

namespace NeoFOAM
{

/**
 * @brief Spalart-Allmaras DDES turbulence model.
 *
 * Registered under the key "SpalartAllmarasDDES" in the TurbulenceModel factory.
 *
 * Two distinct wall-distance fields are required:
 *  - wallDist     : cell-centred distances (Foam::wallDist::y()), for DDES shielding
 *  - nearWallDist : boundary-face distances (turbulenceModel::y()[patchi]), for wall BCs
 *
 * The model owns nuTilda and nut internally.  Call initialize() with disk-read
 * NeoN fields before the first validate() to seed non-zero initial conditions.
 * The backward-compatible three-argument validate(U, nuTilda, nut) and
 * correct(U, phi, nuTilda, nut, rt) overloads are retained for callers that
 * manage those fields externally (e.g. neoPisoFoam).
 */
class SpalartAllmarasDDES : public TurbulenceModel::Register<SpalartAllmarasDDES>
{
public:

    static std::string name() { return "SpalartAllmarasDDES"; }
    static std::string doc() { return "Spalart-Allmaras DDES turbulence model"; }
    static std::string schema() { return "{}"; }

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
     * @brief Factory constructor — reads all turbulence infrastructure from RunTime.
     *
     * Maps the nuTilda solver subdict, then computes wallDist/nearWallDist/delta
     * from the OpenFOAM mesh and reads the initial nuTilda/nut from the current
     * time directory.
     */
    SpalartAllmarasDDES(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu);

    /**
     * @brief Construct from an explicit MeshAdapter (no RunTime/fvSolution access).
     *
     * Computes wallDist, nearWallDist, and delta internally from the mesh, and reads
     * the initial nuTilda/nut fields from the current time directory.
     */
    SpalartAllmarasDDES(
        const NeoN::Executor& exec,
        MeshAdapter& mesh,
        const nnfvcc::VolumeField<scalar>& nu
    );

    /**
     * @brief Backward-compatible constructor for callers that supply prebuilt NeoN fields.
     *
     * @param exec         Kokkos executor (Serial/CPU/GPU)
     * @param mesh         NeoN unstructured mesh
     * @param nu           Laminar kinematic viscosity (cell-centred)
     * @param wallDist     Cell-centred wall distances (Foam::wallDist::y())
     * @param nearWallDist Boundary-face wall distances (nearWallDist::y()[patchi])
     * @param delta        LES filter width (cube root of cell volume or LESModel::delta())
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
    nnfvcc::SurfaceField<scalar>& nuEff() override;

    /// @brief Turbulent viscosity (model-owned; seeded by initialize())
    const nnfvcc::VolumeField<scalar>& nut() const override;

    /// @brief Velocity gradient tensor (updated each correct() call, for viscousStress term)
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const override;

    /// @brief Recompute gradU_ in place at U (+ proc-halo exchange); see TurbulenceModel.
    void updateGradU(const nnfvcc::VolumeField<NeoN::Vec3>& U) override;

    void initialize(
        const nnfvcc::VolumeField<scalar>& nuTildaInit,
        const nnfvcc::VolumeField<scalar>& nutInit
    ) override;

    void validate(const nnfvcc::VolumeField<Vec3>& U) override;

    void correct(const nnfvcc::VolumeField<Vec3>& U, nnfvcc::SurfaceField<scalar>& phi, RunTime& rt)
        override;

    void rotateOldTimes() override;

    void write(MeshAdapter& mesh) const override;

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

    // Registers a zero-seeded "nuTilda" field in db's VectorCollection and returns a reference to
    // the registered copy. initialize() later overwrites it with the disk-read initial condition.
    // Registration is required so the implicit ddt term (oldTime lookup) and rotateOldTimes() can
    // allocate and rotate nuTilda's old-time buffers.
    static nnfvcc::VolumeField<scalar>& registerNuTilda(
        NeoN::Database& db,
        const NeoN::Executor& exec,
        const NeoN::UnstructuredMesh& mesh
    );

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;

    // Laminar viscosity: held by reference — must outlive this object
    const nnfvcc::VolumeField<scalar>& nu_;
    // Physics inputs owned by the model (copy-constructed from caller or computed from OF mesh)
    nnfvcc::VolumeField<scalar> wallDist_;
    nnfvcc::VolumeField<scalar> nearWallDist_;
    nnfvcc::VolumeField<scalar> delta_;

    // Model-owned database hosting nuTilda and its old-time buffers. nuTilda carries an implicit
    // ddt term and is rotated every step, so it MUST be registered: oldTime()/rotateOldTimes()
    // look the field up through field.db()/fieldCollectionName. Registering it in a model-private
    // database keeps the "model owns nuTilda" contract without depending on the solver's field
    // collection. Declared before nuTilda_ so it is alive when nuTilda_ binds to it.
    NeoN::Database turbDb_;

    // Model-owned transport scalars. nuTilda_ references the registered field living in turbDb_'s
    // VectorCollection; nut_ is a plain value (never rotated, no ddt term). Seeded by initialize().
    nnfvcc::VolumeField<scalar>& nuTilda_;
    nnfvcc::VolumeField<scalar> nut_;

    // Cached face-interpolated nu (computed once in constructor)
    nnfvcc::SurfaceField<scalar> surfNu_;

    // Persistent fields carrying state from one correct() call to the next:
    //  - gradU_      : consumed by the momentum predictor before the next correct() runs
    //  - nuEff_      : consumed by the momentum laplacian before the next correct() runs
    //  - nuTildaEff_ : consumed by the nuTilda laplacian at the start of the next correct()
    nnfvcc::VolumeField<NeoN::Tensor> gradU_;
    nnfvcc::SurfaceField<scalar> nuEff_;
    nnfvcc::SurfaceField<scalar> nuTildaEff_;

    // Cached gradient operators (constructed once), runtime-selected from gradSchemes so
    // grad(U) (tensor) and grad(nuTilda) honour the configured scheme (e.g. cellLimited).
    // Default to Gauss-Green when no scheme dictionary is available.
    std::unique_ptr<nnfvcc::GradOperatorFactory<Vec3>> gradUOp_;
    std::unique_ptr<nnfvcc::GradOperatorFactory<Vec3>> gradNuTildaOp_;
    nnfvcc::SurfaceInterpolation<scalar> surfInterp_;

    // SA-DDES model coefficients
    Coefficients coeffs_;
    scalar cw1_;
};

} // namespace NeoFOAM
