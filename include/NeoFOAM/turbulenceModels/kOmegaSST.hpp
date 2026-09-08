// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/auxiliary/bound.hpp"

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
    nnfvcc::SurfaceField<scalar>& dkEff();

    /// @brief Effective omega diffusion coefficient on faces: alphaOmega(F1)*nut + nu
    nnfvcc::SurfaceField<scalar>& domegaEff();

    /// @brief Velocity gradient tensor (updated each correct() call)
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const;

    /// @brief Recompute gradUTmp_ in place at the given velocity (internal + boundary).
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
    const nnfvcc::VolumeField<scalar>& f1Field() const { return F1Tmp_; }
    const nnfvcc::VolumeField<scalar>& pkField() const { return PkTmp_; }
    const nnfvcc::VolumeField<scalar>& spKField() const { return spKTmp_; }
    const nnfvcc::VolumeField<scalar>& omegaSourceField() const { return omegaSourceTmp_; }
    const nnfvcc::VolumeField<scalar>& spOmegaField() const { return spOmegaTmp_; }

    // Physics kernels — public to allow unit-testing of individual steps

    /**
     * @brief Compute sources for k and omega equations plus F1.
     *
     * Uses the OLD nut (current nut, before the PDE solve) for production G = nut * GbyNu0.
     * Writes F1Tmp_, PkTmp_, spKTmp_, omegaSourceTmp_, spOmegaTmp_.
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
     * @brief Update nut internal vector from new k, omega and current gradUTmp_ (S2).
     *
     * Writes only the internal (non-boundary) cells of nut.internalVector().
     */
    void correctNutInternal(
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& omega,
        nnfvcc::VolumeField<scalar>& nut
    ) const;

    /**
     * @brief Recompute surface diffusivity fields from current nut and F1Tmp_.
     *
     * Interpolates nut → surfNutTmp_, F1Tmp_ → surfF1Tmp_, then fills
     * dkEffFTmp_, domegaEffFTmp_, nuEffTmp_.
     */
    void calcDiffusivities(const nnfvcc::VolumeField<scalar>& nut);

    /** @brief Replace the default Gauss-Green tensor-gradient operator with the
     *  gradSchemes-configured one, so grad(U) honours e.g. cellLimited. Used by the
     *  RunTime-constructed wrapper, which alone can reach the schemes dictionary. */
    void setGradUOperator(std::shared_ptr<nnfvcc::GradOperatorFactory<Vec3>> op)
    {
        gradUOp_ = std::move(op);
    }

private:

    void reserveScratch();
    void releaseScratch();
    static bool devicePoolActive();

    // NO-OP WITHOUT A POOL: resize(0)+regrow without a pool is a raw cudaMalloc/cudaFree pair
    // per step — costlier than keeping the field resident.
    template<class... Vs>
    void freeVecs(Vs&... vs)
    {
        if (!devicePoolActive()) return;
        NeoN::fence(exec_);
        auto drop = [](auto& v)
        {
            if (v.size() != 0) v.resize(0);
        };
        (drop(vs), ...);
    }

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;

    // Constant physics inputs (held by reference — must outlive this object)
    const nnfvcc::VolumeField<scalar>& nu_;
    const nnfvcc::VolumeField<scalar>& wallDist_;

    // Owner-cell wall distance copied to boundary faces so wall BCs get a non-zero y.
    nnfvcc::VolumeField<scalar> nearWallDistTmp_;

    // Cached face-interpolated nu (computed once in constructor)
    nnfvcc::SurfaceField<scalar> surfNuTmp_;

    // Intermediate volume fields
    nnfvcc::VolumeField<NeoN::Tensor> gradUTmp_;
    nnfvcc::VolumeField<Vec3> gradKTmp_;
    nnfvcc::VolumeField<Vec3> gradOmegaTmp_;
    nnfvcc::VolumeField<scalar> F1Tmp_;
    nnfvcc::VolumeField<scalar> PkTmp_;
    nnfvcc::VolumeField<scalar> spKTmp_;
    nnfvcc::VolumeField<scalar> omegaSourceTmp_;
    nnfvcc::VolumeField<scalar> spOmegaTmp_;

    // Surface fields
    nnfvcc::SurfaceField<scalar> surfNutTmp_;
    nnfvcc::SurfaceField<scalar> surfF1Tmp_;
    nnfvcc::SurfaceField<scalar> nuEffTmp_;
    nnfvcc::SurfaceField<scalar> dkEffFTmp_;
    nnfvcc::SurfaceField<scalar> domegaEffFTmp_;

    // Cached operators (constructed once)
    // Scalar grad(k)/grad(omega): the factory interface exposes only a Vector<Vec3> output
    // overload, so these stay on GaussGreenGrad until NeoN offers a VolumeField one.
    nnfvcc::GaussGreenGrad gradOp_;
    // gradSchemes-configured tensor-gradient operator, shared with RunTime's gradScheme cache
    std::shared_ptr<nnfvcc::GradOperatorFactory<Vec3>> gradUOp_;
    nnfvcc::SurfaceInterpolation<scalar> surfInterp_;

    // Model coefficients
    Coefficients coeffs_;

    // Per-cell omega wall pin (equivalent of omegaWallFunction::manipulateMatrix).
    NeoN::Vector<scalar> omegaWallValueTmp_;
    NeoN::Vector<scalar> omegaWallMaskTmp_;

    // 1/(N wall faces touching cell) for corner averaging; 0 for non-wall cells.
    NeoN::Vector<scalar> cornerWeightTmp_;
    bool cornerWeightsBuilt_ = false;

    // Lower bound applied to k after each solve, as OpenFOAM's kMin_.
    scalar kMin_ = 0.0;

    // Mesh-derived scratch shared by both bound() calls (it depends only on the mesh).
    mutable BoundCache boundCache_;
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
