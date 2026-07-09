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

    Laminar(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu);

    void validate(const nnfvcc::VolumeField<NeoN::Vec3>& U) override;

    void correct(
        const nnfvcc::VolumeField<NeoN::Vec3>& U,
        nnfvcc::SurfaceField<NeoN::scalar>& phi,
        RunTime& rt
    ) override;

    nnfvcc::SurfaceField<NeoN::scalar>& nuEff() override;

    const nnfvcc::VolumeField<NeoN::scalar>& nut() const override;

    const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const override;

    void updateGradU(const nnfvcc::VolumeField<NeoN::Vec3>& U) override;

    void rotateOldTimes() override {}

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
