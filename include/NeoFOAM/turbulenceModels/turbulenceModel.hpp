// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

/**
 * @brief Abstract base for NeoFOAM turbulence models with runtime selection.
 *
 * Concrete models register via `TurbulenceModel::Register<Derived>` and are
 * instantiated by name through `TurbulenceModel::create(rt, nu)`, which reads
 * the model name from constant/turbulenceProperties.  Models own all transport
 * scalars (nuTilda, nut, …) and map their own solver subdicts in their constructors.
 */
class TurbulenceModel :
    public NeoN::RuntimeSelectionFactory<
        TurbulenceModel,
        NeoN::Parameters<
            RunTime&,
            const nnfvcc::VolumeField<NeoN::scalar>& // nu
            >>
{
public:

    virtual ~TurbulenceModel() = default;

    static std::string name() { return "TurbulenceModel"; }

    /**
     * @brief Read the model name from constant/turbulenceProperties and create the model.
     *
     * Reads simulationType; for LES it also reads the LESModel subkey.
     * Equivalent to `create(modelKey, rt, nu)` but without the solver needing to know
     * which turbulence model is selected.
     */
    static std::unique_ptr<TurbulenceModel>
    create(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu);

    // Optional: seed internal transport scalars from disk-read NeoN fields.
    virtual void initialize(
        const nnfvcc::VolumeField<NeoN::scalar>& nuTildaInit,
        const nnfvcc::VolumeField<NeoN::scalar>& nutInit
    )
    {
        (void)nuTildaInit;
        (void)nutInit;
    }

    // Seed gradU and nut before the time loop.
    virtual void validate(const nnfvcc::VolumeField<NeoN::Vec3>& U) = 0;

    // Update the turbulence model after the PIMPLE loop each time step.
    virtual void correct(
        const nnfvcc::VolumeField<NeoN::Vec3>& U,
        nnfvcc::SurfaceField<NeoN::scalar>& phi,
        RunTime& rt
    ) = 0;

    // Surface effective viscosity (nu + nut) for the momentum laplacian.
    virtual nnfvcc::SurfaceField<NeoN::scalar>& nuEff() = 0;

    // Volume turbulent viscosity for the explicit viscousStress term.
    virtual const nnfvcc::VolumeField<NeoN::scalar>& nut() const = 0;

    // Velocity gradient tensor updated each correct() call.
    virtual const nnfvcc::VolumeField<NeoN::Tensor>& gradU() const = 0;

    // Rotate time-dependent fields for BDF2 time advancement.
    virtual void rotateOldTimes() = 0;

    // Write model-owned fields (nuTilda, nut, …) to disk.
    virtual void write(MeshAdapter& mesh) const = 0;
};

/**
 * @brief Out-of-line wrapper around TurbulenceModel::create.
 *
 * Force-registers the built-in models in libNeoFOAM's runtime-selection table
 * before dispatching, so the factory resolves the model name even when the model
 * is only ever reached through the -fvisibility=hidden Python bindings (where the
 * automatic static self-registration can be elided). addSubType() is idempotent.
 */
std::unique_ptr<TurbulenceModel>
createTurbulenceModel(RunTime& rt, const nnfvcc::VolumeField<NeoN::scalar>& nu);

} // namespace NeoFOAM
