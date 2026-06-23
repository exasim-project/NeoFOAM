// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoFOAM/turbulenceModels/turbulenceModel.hpp"

namespace NeoFOAM
{

/**
 * @brief No-op turbulence model for laminar flow.
 *
 * nut is identically zero; nuEff() returns the surface-interpolated laminar nu.
 * gradU is recomputed each validate()/correct() for use in the viscousStress term.
 */
class Laminar : public TurbulenceModel::Register<Laminar>
{
public:

    static std::string name() { return "laminar"; }
    static std::string doc() { return "No-op laminar model (nut = 0, nuEff = nu)"; }
    static std::string schema() { return "{}"; }

    /** @brief Construct the laminar model; interpolates nu to faces for nuEff. */
    Laminar(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu);

    /** @brief Compute gradU once before the time loop. */
    void validate(const nnfvcc::VolumeField<NeoN::Vec3>& U) override;

    /** @brief Recompute gradU each time step (nut stays zero). */
    void correct(
        const nnfvcc::VolumeField<NeoN::Vec3>& U,
        nnfvcc::SurfaceField<NeoN::scalar>& phi,
        RunTime& rt
    ) override;

    /** @brief Returns the face-interpolated laminar ν (ν_t = 0). */
    nnfvcc::SurfaceField<NeoN::scalar>& nuEff() override;

    /** @brief Returns the zero turbulent viscosity field. */
    const nnfvcc::VolumeField<NeoN::scalar>& nut() const override;

    /** @brief Velocity gradient tensor (updated each correct()). */
    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const override;

    void updateGradU(const nnfvcc::VolumeField<NeoN::Vec3>& U) override;


    void rotateOldTimes() override {}

    /** @brief No-op: laminar model owns no fields to write. */
    void write(MeshAdapter&) const override {}

private:

    NeoN::Executor exec_;
    const NeoN::UnstructuredMesh& mesh_;
    const nnfvcc::VolumeField<NeoN::scalar>& nu_;
    nnfvcc::GaussGreenGrad gradOp_;
    nnfvcc::VolumeField<NeoN::Tensor> gradU_;
    nnfvcc::SurfaceField<NeoN::scalar> nuEff_;
    nnfvcc::VolumeField<NeoN::scalar> nut_;
    nnfvcc::SurfaceInterpolation<NeoN::scalar> surfInterp_;
};

} // namespace NeoFOAM
