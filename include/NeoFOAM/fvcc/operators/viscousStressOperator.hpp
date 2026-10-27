// SPDX-FileCopyrightText: 2024 - 2026 NeoFOAM authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/fvcc/operators/devStress.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/**
 * @brief The divSchemes key of the explicit viscous stress term, as OpenFOAM names it.
 *
 * OpenFOAM derives the key from the expression it discretises: the incompressible solvers have
 * div((nuEff*dev2(T(grad(U))))) and the compressible ones, whose coefficient is rho*nuEff,
 * div(((rho*nuEff)*dev2(T(grad(U))))).
 */
inline std::string viscousStressSchemeKey(bool compressible = false)
{
    return compressible ? "div(((rho*nuEff)*dev2(T(grad(U)))))" : "div((nuEff*dev2(T(grad(U)))))";
}

// ----------------------------
// Operator runtime selection
// ----------------------------

class ViscousStressOperatorFactory :
    public NeoN::RuntimeSelectionFactory<
        ViscousStressOperatorFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{
public:

    static std::unique_ptr<ViscousStressOperatorFactory> create(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        const Input& inputs,
        const std::string& schemeKey = viscousStressSchemeKey()
    )
    {
        std::string key = (std::holds_alternative<NeoN::Dictionary>(inputs))
                            ? std::get<NeoN::Dictionary>(inputs).get<std::string>(schemeKey)
                            : std::get<NeoN::TokenList>(inputs).next<std::string>();

        ViscousStressOperatorFactory::keyExistsOrError(key);
        return ViscousStressOperatorFactory::table().at(key)(exec, mesh, inputs);
    }

    static std::string name() { return "ViscousStressOperatorFactory"; }

    ViscousStressOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec)
        , mesh_(mesh)
    {}

    virtual ~ViscousStressOperatorFactory() = default;

    virtual void explicitOp(
        Vector<Vec3>& rhs,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        const dsl::Coeff operatorScaling
    ) const = 0;

    virtual std::unique_ptr<ViscousStressOperatorFactory> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};


// ----------------------------
// DSL-facing operator wrapper
// ----------------------------

class ViscousStressOperator : public dsl::OperatorMixin<VolumeField<Vec3>, VolumeField<scalar>>
{
public:

    using VectorValueType = Vec3;

    ViscousStressOperator(const ViscousStressOperator& other)
        : dsl::OperatorMixin<VolumeField<Vec3>, VolumeField<scalar>>(
            other.exec_,
            other.coeffs_,
            other.nu_,
            other.type_
        )
        , nu_(other.nu_)
        , nut_(other.nut_)
        , gradUTensor_(other.gradUTensor_)
        , schemeKey_(other.schemeKey_)
        , viscousOp_(other.viscousOp_ ? other.viscousOp_->clone() : nullptr)
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        Input input,
        std::string schemeKey = viscousStressSchemeKey()
    )
        : dsl::OperatorMixin<VolumeField<Vec3>, VolumeField<scalar>>(
            nut.exec(),
            dsl::Coeff(1.0),
            nu,
            termType
        )
        , nu_(nu)
        , nut_(nut)
        , gradUTensor_(&gradU)
        , schemeKey_(std::move(schemeKey))
        , viscousOp_(ViscousStressOperatorFactory::create(this->exec_, nut.mesh(), input, schemeKey_))
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        std::unique_ptr<ViscousStressOperatorFactory> viscousOp,
        std::string schemeKey = viscousStressSchemeKey()
    )
        : dsl::OperatorMixin<VolumeField<Vec3>, VolumeField<scalar>>(
            nut.exec(),
            dsl::Coeff(1.0),
            nu,
            termType
        )
        , nu_(nu)
        , nut_(nut)
        , gradUTensor_(&gradU)
        , schemeKey_(std::move(schemeKey))
        , viscousOp_(std::move(viscousOp))
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        std::string schemeKey = viscousStressSchemeKey()
    )
        : dsl::OperatorMixin<VolumeField<Vec3>, VolumeField<scalar>>(
            nut.exec(),
            dsl::Coeff(1.0),
            nu,
            termType
        )
        , nu_(nu)
        , nut_(nut)
        , gradUTensor_(&gradU)
        , schemeKey_(std::move(schemeKey))
        , viscousOp_(nullptr)
    {}

    void explicitOperation(Vector<Vec3>& source) const
    {
        NF_ASSERT(viscousOp_, "ViscousStressOperatorStrategy not initialized");
        NF_ASSERT(gradUTensor_, "gradU not initialized");
        Vector<Vec3> tmpsource(source.exec(), source.size(), zero<Vec3>());
        const auto operatorScaling = this->getCoefficient();
        viscousOp_->explicitOp(tmpsource, nu_, nut_, *gradUTensor_, operatorScaling);
        source += tmpsource;
    }

    void implicitOperation(la::LinearSystem<Vec3>& ls) const;

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            auto tokens = dict.subDict("divSchemes").get<NeoN::TokenList>(schemeKey_);
            viscousOp_ = ViscousStressOperatorFactory::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            viscousOp_ = ViscousStressOperatorFactory::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "ViscousStressOperator"; }

    // The divSchemes key the operator reads its scheme from, for the scheme defaults the
    // compatibility layer fills in
    Dictionary getConfig() const
    {
        Dictionary config;
        config.insert("schemeKey", schemeKey_);
        return config;
    }

private:

    const VolumeField<scalar>& nu_;
    const VolumeField<scalar>& nut_;
    const VolumeField<Tensor>* gradUTensor_;
    std::string schemeKey_;
    std::unique_ptr<ViscousStressOperatorFactory> viscousOp_;
};


// ----------------------------
// Concrete implementation: Gauss
// ----------------------------

class GaussViscousStress : public ViscousStressOperatorFactory::Register<GaussViscousStress>
{
    using Base = ViscousStressOperatorFactory::Register<GaussViscousStress>;

public:

    static std::string name() { return "Gauss"; }
    static std::string doc() { return "Gauss explicit viscous stress (SymmTensor-based)"; }
    static std::string schema() { return "none"; }

    GaussViscousStress(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh)
        , surfaceInterpolationTensor_(
              exec,
              mesh,
              std::make_unique<Linear<Tensor>>(exec, mesh, Dictionary())
          )
    {}

    void explicitOp(
        Vector<Vec3>& rhs,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        const dsl::Coeff operatorScaling
    ) const override;

    VolumeField<Vec3> viscousStress(
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        const dsl::Coeff operatorScaling
    ) const;

    void viscousStress(
        VolumeField<Vec3>& result,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        const dsl::Coeff operatorScaling
    ) const;

    void viscousStress(
        Vector<Vec3>& result,
        const VolumeField<scalar>& nu,
        const VolumeField<scalar>& nut,
        const VolumeField<Tensor>& gradU,
        const dsl::Coeff operatorScaling
    ) const;

    std::unique_ptr<ViscousStressOperatorFactory> clone() const override
    {
        return std::make_unique<GaussViscousStress>(*this);
    }

private:

    SurfaceInterpolation<Tensor> surfaceInterpolationTensor_;
};

} // namespace NeoN::finiteVolume::cellCentred


// ----------------------------
// DSL free function (extends NeoN::dsl::exp)
// ----------------------------

namespace NeoN::dsl::exp
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

/// compressible: take the scheme from the divSchemes key of the compressible solvers,
/// div(((rho*nuEff)*dev2(T(grad(U))))), instead of the incompressible div((nuEff*dev2(T(grad(U)))))
inline SpatialOperator<Vec3> viscousStress(
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nut,
    const fvcc::VolumeField<Tensor>& gradU,
    bool compressible = false
)
{
    return SpatialOperator<Vec3>(fvcc::ViscousStressOperator(
        dsl::Operator::Type::Explicit, nu, nut, gradU, fvcc::viscousStressSchemeKey(compressible)
    ));
}

} // namespace NeoN::dsl::exp


namespace NeoFOAM
{

/** @brief Build the explicit viscous-stress SpatialOperator inside libNeoFOAM.
 *
 * Thin out-of-line wrapper around ``NeoN::dsl::exp::viscousStress``. The
 * ``ViscousStressOperator``'s assembly-time ``read()`` resolves the "Gauss"
 * strategy through ``ViscousStressOperatorFactory``'s runtime-selection table,
 * which is populated by the ``GaussViscousStress`` self-registration compiled
 * into libNeoFOAM. Instantiating the operator (and its type-erased model thunks)
 * here — rather than inline in a ``-fvisibility=hidden`` Python-binding TU —
 * keeps the factory lookup bound to libNeoFOAM's populated table instead of a
 * private, empty per-module copy.
 */
NeoN::dsl::SpatialOperator<NeoN::Vec3> makeViscousStress(
    const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& nu,
    const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& nut,
    const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::Tensor>& gradU,
    bool compressible = false
);

} // namespace NeoFOAM
