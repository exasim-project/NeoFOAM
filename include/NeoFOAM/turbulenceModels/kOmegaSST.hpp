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
 * @brief NeoFOAM-level wrapper for the k-omega SST turbulence model (Menter 1993/2003).
 *
 * Mirrors the OpenFOAM turbulence->validate() / turbulence->correct() interface.
 * Owns all intermediate fields and cached operators.  The solver manages the primary
 * fields (k, omega, nut) and calls validate() once at setup and correct() after each
 * momentum solve.
 *
 * Coefficients follow the 2003 Menter/Kuntz/Langtry values (same as OpenFOAM default).
 * Diffusion is written in terms of alpha = 1/sigma so that the inner/outer blending
 * (F1) is applied uniformly — consistent with the OpenFOAM kOmegaSSTBase convention.
 */
class KOmegaSST
{
public:

    struct Coefficients
    {
        scalar alphaK1 = 0.85;
        scalar alphaK2 = 1.0;
        scalar alphaOmega1 = 0.5;
        scalar alphaOmega2 = 0.856;
        scalar gamma1 = 5.0 / 9.0;
        scalar gamma2 = 0.44;
        scalar beta1 = 0.075;
        scalar beta2 = 0.0828;
        scalar betaStar = 0.09;
        scalar a1 = 0.31;
        scalar b1 = 1.0;
        scalar c1 = 10.0;
    };

    /**
     * @brief Construct the turbulence model wrapper.
     *
     * @param exec     Kokkos executor (Serial/CPU/GPU)
     * @param mesh     NeoN unstructured mesh
     * @param nu       Laminar kinematic viscosity (cell-centred)
     * @param wallDist Cell-centred wall distances (Foam::wallDist::y())
     */
    KOmegaSST(
        const NeoN::Executor& exec,
        const NeoN::UnstructuredMesh& mesh,
        const nnfvcc::VolumeField<scalar>& nu,
        const nnfvcc::VolumeField<scalar>& wallDist
    );

    /**
     * @brief Initialise nut from the current k and omega.  Call once at setup.
     *
     * Equivalent to OpenFOAM's turbulence->validate().
     *
     * @param U     Initial velocity field
     * @param k     Turbulent kinetic energy (read-only)
     * @param omega Specific dissipation rate (read-only)
     * @param nut   Turbulent viscosity (updated in place)
     */
    void validate(
        const nnfvcc::VolumeField<Vec3>& U,
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& omega,
        nnfvcc::VolumeField<scalar>& nut
    );

    /**
     * @brief Full turbulence update: solve k and omega PDEs, update nut.
     *
     * Equivalent to OpenFOAM's turbulence->correct().
     *
     * @param U     Current velocity field
     * @param phi   Face flux field
     * @param k     Turbulent kinetic energy (updated in place)
     * @param omega Specific dissipation rate (updated in place)
     * @param nut   Turbulent viscosity (updated in place)
     * @param rt    NeoFOAM RunTime
     */
    void correct(
        const nnfvcc::VolumeField<Vec3>& U,
        nnfvcc::SurfaceField<scalar>& phi,
        nnfvcc::VolumeField<scalar>& k,
        nnfvcc::VolumeField<scalar>& omega,
        nnfvcc::VolumeField<scalar>& nut,
        RunTime& rt
    );

    /// @brief Effective viscosity on faces: nu + nut (for momentum equation laplacian)
    nnfvcc::SurfaceField<scalar>& nuEff();

    /// @brief Effective k diffusion coefficient on faces: alphaK(F1)*nut + nu
    nnfvcc::SurfaceField<scalar>& DkEff();

    /// @brief Effective omega diffusion coefficient on faces: alphaOmega(F1)*nut + nu
    nnfvcc::SurfaceField<scalar>& DomegaEff();

    /// @brief Velocity gradient tensor (updated each correct() call)
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const;

    /// @brief Recompute gradU_ in place at the given velocity (internal + boundary).
    void updateGradU(const nnfvcc::VolumeField<Vec3>& U);

    /**
     * @brief Returns the deviatoric stress at all boundary faces as Vector<SymmTensor>.
     *
     * Computes -nuEff * symm(twoSymm(gradU)).dev2() at each boundary face.
     */
    NeoN::Vector<NeoN::SymmTensor> devRhoReff() const;

    /// @brief Read-only access to model coefficients
    const Coefficients& coeffs() const { return coeffs_; }

    // Internal result fields — read-only access for testing/inspection
    const nnfvcc::VolumeField<scalar>& F1Field() const { return F1_; }
    const nnfvcc::VolumeField<scalar>& PkField() const { return Pk_; }
    const nnfvcc::VolumeField<scalar>& spKField() const { return spK_; }
    const nnfvcc::VolumeField<scalar>& omegaSourceField() const { return omegaSource_; }
    const nnfvcc::VolumeField<scalar>& spOmegaField() const { return spOmega_; }

    // Physics kernels — public to allow unit-testing of individual steps

    /**
     * @brief Compute sources for k and omega equations plus F1.
     *
     * Uses the OLD nut (current nut, before the PDE solve) for production G = nut * GbyNu0.
     * Writes F1_, Pk_, spK_, omegaSource_, spOmega_.
     */
    void computeF1AndSources(
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& omega,
        const nnfvcc::VolumeField<scalar>& nut,
        const nnfvcc::VolumeField<Vec3>& gradK,
        const nnfvcc::VolumeField<Vec3>& gradOmega,
        const nnfvcc::VolumeField<NeoN::Tensor>& gradU
    );

    /**
     * @brief Update nut internal vector from new k, omega and current gradU_ (S2).
     *
     * Writes only the internal (non-boundary) cells of nut.internalVector().
     */
    void correctNutInternal(
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& omega,
        nnfvcc::VolumeField<scalar>& nut
    ) const;

    /**
     * @brief Recompute surface diffusivity fields from current nut and F1_.
     *
     * Interpolates nut → surfNut_, F1_ → surfF1_, then fills
     * DkEffF_, DomegaEffF_, nuEff_.
     */
    void calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut);

private:

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;

    // Constant physics inputs (held by reference — must outlive this object)
    const nnfvcc::VolumeField<scalar>& nu_;
    const nnfvcc::VolumeField<scalar>& wallDist_;

    // Per-boundary-face cell-to-wall distance — populated in the constructor
    // by copying wallDist_'s owner-cell internal value into each boundary
    // face's slot. Inserted into BoundaryContext under the name "nearWallDist"
    // so omegaWallFunction / nutUSpaldingWallFunction get a non-zero y at the
    // wall (wallDist_'s own boundary values are ≈ 0 on a wall patch, which
    // would let ω_vis = 6ν/(β₁y²) explode).
    nnfvcc::VolumeField<scalar> nearWallDist_;

    // Cached face-interpolated nu (computed once in constructor)
    nnfvcc::SurfaceField<scalar> surfNu_;

    // Intermediate volume fields
    nnfvcc::VolumeField<NeoN::Tensor> gradU_;
    nnfvcc::VolumeField<Vec3> gradK_;
    nnfvcc::VolumeField<Vec3> gradOmega_;
    nnfvcc::VolumeField<scalar> F1_;
    nnfvcc::VolumeField<scalar> Pk_;
    nnfvcc::VolumeField<scalar> spK_;
    nnfvcc::VolumeField<scalar> omegaSource_;
    nnfvcc::VolumeField<scalar> spOmega_;

    // Surface fields
    nnfvcc::SurfaceField<scalar> surfNut_;
    nnfvcc::SurfaceField<scalar> surfF1_;
    nnfvcc::SurfaceField<scalar> nuEff_;
    nnfvcc::SurfaceField<scalar> DkEffF_;
    nnfvcc::SurfaceField<scalar> DomegaEffF_;

    // Cached operators (constructed once)
    nnfvcc::GaussGreenGrad gradOp_;
    nnfvcc::SurfaceInterpolation<scalar> surfInterp_;

    // Model coefficients
    Coefficients coeffs_;

    // Per-cell omega wall-function constraint, rebuilt each correct(): omegaWallMask_[c] != 0
    // marks a wall-adjacent cell whose omega is hard-pinned to omegaWallValue_[c] (the blended
    // viscous/log value) when solving the omega equation — the equivalent of OpenFOAM's
    // omegaWallFunction::manipulateMatrix(setValues). Without this pin near-wall omega stays
    // too small and nut blows up. Sized nCells, owned here (passed by ref to PDE::setConstraints).
    NeoN::Vector<scalar> omegaWallValue_;
    NeoN::Vector<scalar> omegaWallMask_;

    // Per-cell corner weight 1/(number of omegaWallFunction faces touching the cell), 0 for
    // non-wall cells. Mirrors OpenFOAM omegaWallFunction::createAveragingWeights: a cell touched
    // by N wall faces averages its N per-face omega/G contributions (weight 1/N each) instead of
    // the previous nondeterministic "last face wins". Built once (the wall topology is static);
    // the gate below guards that one-time fill.
    NeoN::Vector<scalar> cornerWeight_;
    bool cornerWeightsBuilt_ = false;

    // Scratch + cache for the smoother lower-bound repair on omega/k (mirrors OpenFOAM's
    // Foam::bound()): boundFloored_/surfBoundFloored_ hold max(field, lowerBound) and its face
    // interpolation; sumFaceArea_ caches the per-cell sum of face areas (the fvc::average
    // denominator surfaceSum(magSf), which is mesh-static so it is built once). Reused by both the
    // omega and k bound() calls in correct().
    nnfvcc::VolumeField<scalar> boundFloored_;
    nnfvcc::SurfaceField<scalar> surfBoundFloored_;
    NeoN::Vector<scalar> sumFaceArea_;
    bool sumFaceAreaBuilt_ = false;
};

/**
 * @brief TurbulenceModel-factory wrapper for k-omega SST.
 *
 * Owns k, omega and nut; reads them from disk in the constructor, delegates
 * physics to KOmegaSST.  Registered as "kOmegaSST" in the TurbulenceModel
 * factory so TurbulenceModel::create() can instantiate it from a RAS
 * turbulenceProperties block.
 */
class KOmegaSSTModel : public TurbulenceModel::Register<KOmegaSSTModel>
{
public:

    static std::string name() { return "kOmegaSST"; }
    static std::string doc() { return "k-omega SST turbulence model"; }
    static std::string schema() { return "{}"; }

    /** @brief Read k, omega, nut from RunTime and construct the KOmegaSST physics object. */
    KOmegaSSTModel(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu);

    /** @brief Initialise nut and diffusivities from the on-disk k/omega. */
    void validate(const nnfvcc::VolumeField<Vec3>& U) override;

    /** @brief Solve the k and omega PDEs and update nut. */
    void correct(const nnfvcc::VolumeField<Vec3>& U, nnfvcc::SurfaceField<scalar>& phi, RunTime& rt)
        override;

    /** @brief Surface effective viscosity (ν + ν_t). */
    nnfvcc::SurfaceField<scalar>& nuEff() override;
    /** @brief Cell-centred turbulent viscosity. */
    const nnfvcc::VolumeField<scalar>& nut() const override;
    /** @brief Velocity gradient tensor (updated each correct()). */
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const override;
    /** @brief Recompute gradU in place at the given velocity. */
    void updateGradU(const nnfvcc::VolumeField<Vec3>& U) override;

    /** @brief Write k, omega and nut fields to disk. */
    void write(MeshAdapter& mesh) const override;

    /** @brief Rotate k, omega and nut old-time levels for BDF2. */
    void rotateOldTimes() override;

private:

    const nnfvcc::VolumeField<scalar>& nu_;
    nnfvcc::VolumeField<scalar> wallDist_;
    nnfvcc::VolumeField<scalar>* k_ = nullptr;
    nnfvcc::VolumeField<scalar>* omega_ = nullptr;
    nnfvcc::VolumeField<scalar>* nut_ = nullptr;
    KOmegaSST model_;
};


} // namespace NeoFOAM
