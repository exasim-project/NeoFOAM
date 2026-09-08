// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

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
 * @brief NeoFOAM-level wrapper for the standard k-ε turbulence model
 * (Launder & Spalding 1974).
 *
 * Mirrors the OpenFOAM turbulence->validate() / turbulence->correct() interface
 * and follows the same layout as KOmegaSST. Owns all intermediate fields and
 * cached operators; the solver manages the primary fields (k, ε, ν_t) and
 * calls validate() once at setup and correct() after each momentum solve.
 *
 * Equations:
 *
 *   ν_t = Cμ k²/ε
 *
 *   ∂k/∂t + ∇·(φ k) - ∇·((ν + ν_t/σ_k) ∇k)   = G - ε
 *   ∂ε/∂t + ∇·(φ ε) - ∇·((ν + ν_t/σ_ε) ∇ε) = (C1 G - C2 ε) ε/k
 *
 * Coefficients are the upstream defaults from `kEpsilon.C` (Cμ=0.09, C1=1.44,
 * C2=1.92, σ_k=1.0, σ_ε=1.3).
 */
class KEpsilon
{
public:

    struct Coefficients
    {
        scalar cmu = 0.09;
        scalar C1 = 1.44;
        scalar C2 = 1.92;
        scalar sigmaK = 1.0;
        scalar sigmaEps = 1.3;
    };

    /**
     * @brief Construct the turbulence model wrapper.
     *
     * @param exec     Kokkos executor (Serial/CPU/GPU)
     * @param mesh     NeoN unstructured mesh
     * @param nu       Laminar kinematic viscosity (cell-centred)
     * @param wallDist Cell-centred wall distances (Foam::wallDist::y())
     */
    KEpsilon(
        const NeoN::Executor& exec,
        const NeoN::UnstructuredMesh& mesh,
        const nnfvcc::VolumeField<scalar>& nu,
        const nnfvcc::VolumeField<scalar>& wallDist
    );

    /**
     * @brief Initialise ν_t from the current k and ε. Call once at setup.
     */
    void validate(
        const nnfvcc::VolumeField<Vec3>& U,
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& epsilon,
        nnfvcc::VolumeField<scalar>& nut
    );

    /**
     * @brief Full turbulence update: solve ε and k PDEs, update ν_t.
     */
    void correct(
        const nnfvcc::VolumeField<Vec3>& U,
        nnfvcc::SurfaceField<scalar>& phi,
        nnfvcc::VolumeField<scalar>& k,
        nnfvcc::VolumeField<scalar>& epsilon,
        nnfvcc::VolumeField<scalar>& nut,
        RunTime& rt
    );

    /// @brief Effective viscosity on faces: ν + ν_t (for momentum laplacian)
    nnfvcc::SurfaceField<scalar>& nuEff();

    /// @brief Effective k diffusion coefficient on faces: ν_t/σ_k + ν
    nnfvcc::SurfaceField<scalar>& dkEff();

    /// @brief Effective ε diffusion coefficient on faces: ν_t/σ_ε + ν
    nnfvcc::SurfaceField<scalar>& depsilonEff();

    /// @brief Velocity gradient tensor (updated each correct() call)
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const;

    /// @brief Recompute gradU_ in place at the given velocity (internal + boundary).
    void updateGradU(const nnfvcc::VolumeField<Vec3>& U);

    /// @brief Returns the deviatoric stress at boundary faces: -ν_eff · symm(twoSymm(∇U)).dev2
    NeoN::Vector<NeoN::SymmTensor> devRhoReff() const;

    /// @brief Read-only access to model coefficients
    const Coefficients& coeffs() const { return coeffs_; }

    // Internal result fields — read-only access for testing
    const nnfvcc::VolumeField<scalar>& pkField() const { return Pk_; }
    const nnfvcc::VolumeField<scalar>& spKField() const { return spK_; }
    const nnfvcc::VolumeField<scalar>& epsilonSourceField() const { return epsilonSource_; }
    const nnfvcc::VolumeField<scalar>& spEpsilonField() const { return spEpsilon_; }

    // Physics kernels — public to allow unit-testing of individual steps

    /**
     * @brief Compute sources for k and ε equations.
     *
     * Uses the OLD ν_t (current ν_t, before the PDE solve) for production G = ν_t·GbyNu0.
     * Writes Pk_, spK_, epsilonSource_, spEpsilon_.
     */
    void computeSources(
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& epsilon,
        const nnfvcc::VolumeField<scalar>& nut,
        const nnfvcc::VolumeField<NeoN::Tensor>& gradU
    );

    /**
     * @brief Update ν_t internal vector from new k, ε: ν_t = Cμ·k²/ε.
     */
    void correctNutInternal(
        const nnfvcc::VolumeField<scalar>& k,
        const nnfvcc::VolumeField<scalar>& epsilon,
        nnfvcc::VolumeField<scalar>& nut
    ) const;

    /**
     * @brief Recompute surface diffusivity fields from current ν_t.
     *
     * Interpolates ν_t → surfNut_, then fills nuEff_, dkEffF_, depsilonEffF_.
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

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;

    // Lower bounds applied to k and epsilon after each solve, as OpenFOAM's kMin_/epsilonMin_.
    // They floor the field; cells that undershoot to zero or below are refilled from the
    // neighbourhood by bound() rather than pinned here.
    scalar kMin_ = 0.0;
    scalar epsilonMin_ = 1e-10;

    // Mesh-derived scratch shared by both bound() calls (it depends only on the mesh).
    mutable BoundCache boundCache_;

    // Constant physics inputs (held by reference — must outlive this object)
    const nnfvcc::VolumeField<scalar>& nu_;
    const nnfvcc::VolumeField<scalar>& wallDist_;

    // Per-boundary-face cell-to-wall distance — populated in the constructor
    // by copying wallDist_'s owner-cell internal value into each boundary
    // face's slot. Used by the epsilon/nut wall-function BCs as "y".
    nnfvcc::VolumeField<scalar> nearWallDist_;

    // Cached face-interpolated nu (computed once in constructor)
    nnfvcc::SurfaceField<scalar> surfNu_;

    // Intermediate volume fields
    nnfvcc::VolumeField<NeoN::Tensor> gradU_;
    nnfvcc::VolumeField<scalar> Pk_;
    nnfvcc::VolumeField<scalar> spK_;
    nnfvcc::VolumeField<scalar> epsilonSource_;
    nnfvcc::VolumeField<scalar> spEpsilon_;

    // Surface fields
    nnfvcc::SurfaceField<scalar> surfNut_;
    nnfvcc::SurfaceField<scalar> nuEff_;
    nnfvcc::SurfaceField<scalar> dkEffF_;
    nnfvcc::SurfaceField<scalar> depsilonEffF_;

    // Cached operators (constructed once)
    // gradSchemes-configured tensor-gradient operator, shared with RunTime's gradScheme cache
    std::shared_ptr<nnfvcc::GradOperatorFactory<Vec3>> gradUOp_;
    nnfvcc::SurfaceInterpolation<scalar> surfInterp_;

    // Model coefficients
    Coefficients coeffs_;

    // Corner-averaging weights for wall-function G feedback (1/N per wall cell, 0 elsewhere).
    // Built once on first correct() since wall topology is static.
    NeoN::Vector<scalar> cornerWeight_;
    bool cornerWeightsBuilt_ = false;
};

/**
 * @brief TurbulenceModel-factory wrapper for k-epsilon.
 *
 * Owns k, epsilon and nut; reads them from disk in the constructor, delegates
 * physics to KEpsilon.  Registered as "kEpsilon" in the TurbulenceModel
 * factory so TurbulenceModel::create() can instantiate it from a RAS
 * turbulenceProperties block.
 */
class KEpsilonModel : public TurbulenceModel::Register<KEpsilonModel>
{
public:

    static std::string name() { return "kEpsilon"; }
    static std::string doc() { return "k-epsilon turbulence model"; }
    static std::string schema() { return "{}"; }

    /** @brief Read k, epsilon, nut from RunTime and construct the KEpsilon physics object. */
    KEpsilonModel(RunTime& rt, const nnfvcc::VolumeField<scalar>& nu);

    /** @brief Initialise ν_t and diffusivities from the on-disk k/ε. */
    void validate(const nnfvcc::VolumeField<Vec3>& U) override;

    /** @brief Solve the ε and k PDEs and update ν_t. */
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

    /** @brief Write k, epsilon and nut fields to disk. */
    void write(MeshAdapter& mesh) const override;

    /** @brief Rotate k, epsilon and nut old-time levels for BDF2. */
    void rotateOldTimes() override;

private:

    const nnfvcc::VolumeField<scalar>& nu_;
    nnfvcc::VolumeField<scalar> wallDist_;
    nnfvcc::VolumeField<scalar>* k_ = nullptr;
    nnfvcc::VolumeField<scalar>* epsilon_ = nullptr;
    nnfvcc::VolumeField<scalar>* nut_ = nullptr;
    KEpsilon model_;
};

} // namespace NeoFOAM
