// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// MRF (multiple reference frame) rotating zones for the NeoN solver family.
//
// Design (report/mrf-modelspec-neon-plan.md §2): do NOT port Foam::MRFZone to
// NeoN — NeoN's mesh has no cell zones. Instead let OpenFOAM's own MRFZoneList
// do the zone bookkeeping on the OpenFOAM mesh and move the *results* into NeoN,
// the precedent set by build_near_wall_dist (wallFunctions.cpp).
//
// Every MRF operation the pressure-velocity algorithms need is either constant
// or linear in U, so the zone list collapses to four constant fields built once
// plus one per-iteration cross product:
//
//   omega_            VolumeField<Vec3>    Omega per cell, zero outside the zones
//   frameFlux_        SurfaceField<scalar> (Omega x r) . Sf, zero off-zone
//   relativeKeep_     SurfaceField<scalar> 0 on the patch faces makeRelative ZEROES
//   zeroFilterMask_   SurfaceField<scalar> 0 on every MRF face (MRF::zeroFilter)
//   boundaryVelocity_ VolumeField<Vec3>    Omega x r on the rotating wall faces
//   boundaryKeep_     VolumeField<scalar>  0 on those same faces, 1 elsewhere
//
// All six are *probed* out of MRFZoneList — each operation applied to a field of
// known constant value and the result read back — which inherits OpenFOAM's exact
// treatment of zone-internal, included and excluded faces and needs no access to
// MRFZone's private face lists.
//
// Exact only because each bound operation is affine with a build-time-fixed mask:
// out = keep * (in - offset), so a zero probe recovers offset and a unit probe
// recovers keep. Outside that invariant, and not to be reached by extending the
// pattern: makeRelative(volVectorField&) (its offset varies per cell and no probe
// carries it there), a time-varying omega (the constructor rejects one), and a
// topology change (all six describe the construction-time mesh, and this path has
// no MRFZone::update()).
//
// Note also that makeAbsolute is not the inverse of makeRelative: it *adds* on the
// included patch faces where makeRelative *assigns* 0 (MRFZoneTemplates.C:186-192
// vs :88-91), so frameFlux_ is 0 exactly where an absolute reconstruction needs it
// non-zero. Binding makeAbsolute needs its own probe through makeAbsolute.

#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// NeoN headers
#include "NeoN/NeoN.hpp"
#include "NeoN/dsl/explicit.hpp"

// OpenFOAM headers
#include "Function1.H"
#include "IOMRFZoneList.H"
#include "volFields.H"
#include "surfaceFields.H"

// NeoFOAM headers
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/readers.hpp" // fromFoamField (Foam field -> NeoN field)

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vec3;

namespace
{

// --------------------------------------------------------------------------
// Foam -> NeoN transfer, calculated boundaries throughout
//
// The probe fields carry no physical boundary condition: only their *values*
// are read, so the NeoN twins get calculated BCs (a no-op correct) rather than
// the type mapping constructFrom would apply.
// --------------------------------------------------------------------------

// Flatten a boundary field into the patch/face order the NeoN boundary ranges
// expect: non-processor patches first, then the processor tail.
template<class FoamValueType, class BoundaryFieldType>
Foam::Field<FoamValueType> flattenBoundary(const BoundaryFieldType& bField, localIdx nBnd)
{
    Foam::Field<FoamValueType> bval(static_cast<Foam::label>(nBnd), Foam::Zero);
    Foam::label bi = 0;
    for (const bool processorPass : {false, true})
    {
        forAll(bField, patchi)
        {
            const auto& pin = bField[patchi];
            if ((pin.patch().type() == "processor") != processorPass)
            {
                continue;
            }
            forAll(pin, i)
            {
                bval[bi++] = pin[i];
            }
        }
    }
    NF_ASSERT_EQUAL(bi, static_cast<Foam::label>(nBnd));
    return bval;
}

fvcc::VolumeField<Vec3> toNeoNVolume(const nf::RunTime& rt, const Foam::volVectorField& in)
{
    fvcc::VolumeField<Vec3> out(
        rt.exec,
        in.name(),
        rt.nfMesh,
        fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh)
    );
    out.internalVector() = nf::fromFoamField(rt.exec, in.primitiveField());
    const localIdx nBnd = out.boundaryData().value().size();
    out.boundaryData().value() =
        nf::fromFoamField(rt.exec, flattenBoundary<Foam::vector>(in.boundaryField(), nBnd));
    return out;
}

fvcc::VolumeField<scalar> toNeoNVolume(const nf::RunTime& rt, const Foam::volScalarField& in)
{
    fvcc::VolumeField<scalar> out(
        rt.exec,
        in.name(),
        rt.nfMesh,
        fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(rt.nfMesh)
    );
    out.internalVector() = nf::fromFoamField(rt.exec, in.primitiveField());
    const localIdx nBnd = out.boundaryData().value().size();
    out.boundaryData().value() =
        nf::fromFoamField(rt.exec, flattenBoundary<Foam::scalar>(in.boundaryField(), nBnd));
    return out;
}

fvcc::SurfaceField<scalar> toNeoNSurface(const nf::RunTime& rt, const Foam::surfaceScalarField& in)
{
    fvcc::SurfaceField<scalar> out(
        rt.exec,
        in.name(),
        rt.nfMesh,
        fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(rt.nfMesh)
    );
    const localIdx nInt = out.internalVector().size();
    Foam::scalarField internalData(static_cast<Foam::label>(nInt));
    for (Foam::label facei = 0; facei < static_cast<Foam::label>(nInt); ++facei)
    {
        internalData[facei] = in[facei];
    }
    const localIdx nBnd = out.boundaryData().value().size();
    out.internalVector() = nf::fromFoamField(rt.exec, internalData);
    out.boundaryData().value() =
        nf::fromFoamField(rt.exec, flattenBoundary<Foam::scalar>(in.boundaryField(), nBnd));
    return out;
}

Foam::volVectorField uniformVolVector(const Foam::fvMesh& mesh, const Foam::vector& value)
{
    return Foam::volVectorField(
        Foam::IOobject(
            "MRF:probe",
            mesh.time().timeName(),
            mesh,
            Foam::IOobjectOption(
                Foam::IOobjectOption::NO_READ,
                Foam::IOobjectOption::NO_WRITE,
                false
            )
        ),
        mesh,
        Foam::dimensionedVector(Foam::dimVelocity, value)
    );
}

Foam::surfaceScalarField uniformSurfaceScalar(const Foam::fvMesh& mesh, Foam::scalar value)
{
    return Foam::surfaceScalarField(
        Foam::IOobject(
            "MRF:probe",
            mesh.time().timeName(),
            mesh,
            Foam::IOobjectOption(
                Foam::IOobjectOption::NO_READ,
                Foam::IOobjectOption::NO_WRITE,
                false
            )
        ),
        mesh,
        Foam::dimensionedScalar(Foam::dimVolume / Foam::dimTime, value)
    );
}

// MRFZone re-evaluates its omega Function1 on every Omega() call, but the frame
// fields below are probed once, at construction, so a table/ramp/sine omega
// would silently run at its construction-time value forever. MRFZone exposes
// neither omega_ nor its coeffs dictionary, so the Function1 is rebuilt from the
// very entry IOMRFZoneList (an IOdictionary) read it from and asked its type.
void rejectTimeVaryingOmega(const Foam::fvMesh& mesh, const Foam::IOMRFZoneList& mrf)
{
    const Foam::MRFZoneList& zones = mrf;
    for (const Foam::MRFZone& zone : zones)
    {
        // An inactive zone never parses omega, and need not carry one.
        if (!zone.active())
        {
            continue;
        }
        const Foam::dictionary& coeffs = mrf.subDict(zone.name());
        if (!Foam::Function1<Foam::scalar>::New("omega", coeffs, &mesh)->constant())
        {
            // invalid_argument, not runtime_error: nanobind maps it to ValueError,
            // which is what the Python-side guard in mrf.py raises for the same case.
            throw std::invalid_argument(
                "MRF zone '" + std::string(zone.name())
                + "': omega is not constant in time, which the NeoN rotating-frame model "
                  "does not support — its frame fields are probed out of the zone list "
                  "once, at construction, so a time-varying omega would be frozen at its "
                  "construction-time value. Use a constant omega."
            );
        }
    }
}

// nvcc refuses an extended device lambda inside a private member, so the two
// face kernels are free functions rather than helpers on the class below.
void takeRelative(
    const NeoN::Executor& exec,
    NeoN::Vector<scalar>& target,
    const NeoN::Vector<scalar>& keepVec,
    const NeoN::Vector<scalar>& fluxVec
)
{
    auto out = target.view();
    auto keep = keepVec.view();
    auto flux = fluxVec.view();
    NeoN::parallelFor(
        exec,
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) { out[i] = keep[i] * (out[i] - flux[i]); },
        "mrfMakeRelative"
    );
}

void maskOut(
    const NeoN::Executor& exec,
    NeoN::Vector<scalar>& target,
    const NeoN::Vector<scalar>& maskVec
)
{
    auto out = target.view();
    auto mask = maskVec.view();
    NeoN::parallelFor(
        exec,
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) { out[i] = mask[i] * out[i]; },
        "mrfZeroFilter"
    );
}

/* @brief MRFZoneList's rotating-frame terms, transferred onto the NeoN mesh.
 *
 * Owns the OpenFOAM zone list, the constant fields probed out of it, and the
 * Coriolis buffer the momentum term is a reference to (NeoN's dsl operators
 * hold their coefficient field by reference, so it must outlive the assembly —
 * it lives here, alongside the run's other model runtimes).
 */
class MRFNeoN
{
public:

    explicit MRFNeoN(nf::RunTime& rt)
        : mrf_(rt.mesh)
        , omega_(buildOmega(rt, mrf_))
        , frameFlux_(buildFrameFlux(rt, mrf_))
        , relativeKeep_(buildRelativeKeep(rt, mrf_))
        , zeroFilterMask_(buildZeroFilterMask(rt, mrf_))
        , boundaryVelocity_(buildBoundaryVelocity(rt, mrf_))
        , boundaryKeep_(buildBoundaryKeep(rt, mrf_))
        , acceleration_(
              rt.exec,
              "MRF:acceleration",
              rt.nfMesh,
              fvcc::createCalculatedBCs<fvcc::VolumeBoundary<Vec3>>(rt.nfMesh)
          )
    {
        rejectTimeVaryingOmega(rt.mesh, mrf_);
    }

    /* @brief MRFZoneList::correctBoundaryVelocity — Omega x r on the rotating
     * wall faces, every other boundary face left as its own BC set it.
     *
     * ``refValue`` is written alongside ``value``, and that is the load-bearing
     * half: NeoN assembles a boundary face from ``valueFraction * refValue``
     * (gaussGreenDiv.cpp, gaussGreenLaplacian.cpp), so a rotating wall written
     * only into ``value`` would still enter the momentum matrix as the case's
     * stationary noSlip. This is what OpenFOAM's ``operator==`` on a fixedValue
     * patch does — it replaces the Dirichlet value, not just the face value. */
    void correctBoundaryVelocity(fvcc::VolumeField<Vec3>& U) const
    {
        auto u = U.boundaryData().value().view();
        auto refValue = U.boundaryData().refValue().view();
        auto keep = boundaryKeep_.boundaryData().value().view();
        auto vel = boundaryVelocity_.boundaryData().value().view();
        NeoN::parallelFor(
            U.exec(),
            {0, u.size()},
            NEON_LAMBDA(const localIdx i) {
                u[i] = keep[i] * u[i] + vel[i];
                refValue[i] = keep[i] * refValue[i] + vel[i];
            },
            "mrfCorrectBoundaryVelocity"
        );
    }

    const fvcc::VolumeField<Vec3>& omega() const { return omega_; }

    const fvcc::SurfaceField<scalar>& frameFlux() const { return frameFlux_; }

    const fvcc::SurfaceField<scalar>& relativeKeep() const { return relativeKeep_; }

    /* @brief MRFZoneList::makeRelative(phi) — subtract the frame flux, and zero
     * the included (rotating) patch faces, which native assigns rather than
     * subtracts. */
    void makeRelative(fvcc::SurfaceField<scalar>& phi) const
    {
        takeRelative(
            phi.exec(),
            phi.internalVector(),
            relativeKeep_.internalVector(),
            frameFlux_.internalVector()
        );
        takeRelative(
            phi.exec(),
            phi.boundaryData().value(),
            relativeKeep_.boundaryData().value(),
            frameFlux_.boundaryData().value()
        );
    }

    /* @brief MRFZoneList::zeroFilter — zero an absolute-frame flux inside the
     * MRF region. */
    fvcc::SurfaceField<scalar> zeroFilter(const fvcc::SurfaceField<scalar>& corr) const
    {
        fvcc::SurfaceField<scalar> out(corr);
        maskOut(out.exec(), out.internalVector(), zeroFilterMask_.internalVector());
        maskOut(out.exec(), out.boundaryData().value(), zeroFilterMask_.boundaryData().value());
        return out;
    }

    /* @brief MRFZoneList::DDt(U) — the frame acceleration Omega x U, zero
     * outside the zones. Refreshes and returns the buffer. */
    const fvcc::VolumeField<Vec3>& acceleration(const fvcc::VolumeField<Vec3>& U)
    {
        auto acc = acceleration_.internalVector().view();
        auto u = U.internalVector().view();
        auto omega = omega_.internalVector().view();
        NeoN::parallelFor(
            acceleration_.exec(),
            {0, acc.size()},
            NEON_LAMBDA(const localIdx i) {
                acc[i] = Vec3(
                    omega[i][1] * u[i][2] - omega[i][2] * u[i][1],
                    omega[i][2] * u[i][0] - omega[i][0] * u[i][2],
                    omega[i][0] * u[i][1] - omega[i][1] * u[i][0]
                );
            },
            "mrfAcceleration"
        );
        return acceleration_;
    }

    /* @brief The momentum term ``+ MRF.DDt(U)`` of UEqn.H, as an explicit
     * source over the refreshed acceleration buffer. */
    NeoN::dsl::SpatialOperator<Vec3> DDt(const fvcc::VolumeField<Vec3>& U)
    {
        acceleration(U);
        return NeoN::dsl::exp::source(acceleration_);
    }

private:

    // Omega per cell, recovered from three unit-velocity probes: MRFZoneList
    // returns Omega x U on the zone cells and Zero elsewhere, so
    //   DDt(ex).y = Omega.z,  DDt(ey).z = Omega.x,  DDt(ez).x = Omega.y.
    static fvcc::VolumeField<Vec3> buildOmega(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        const Foam::fvMesh& mesh = rt.mesh;
        Foam::volVectorField probeX(uniformVolVector(mesh, Foam::vector(1, 0, 0)));
        Foam::volVectorField probeY(uniformVolVector(mesh, Foam::vector(0, 1, 0)));
        Foam::volVectorField probeZ(uniformVolVector(mesh, Foam::vector(0, 0, 1)));
        const Foam::vectorField aX(mrf.DDt(probeX)().primitiveField());
        const Foam::vectorField aY(mrf.DDt(probeY)().primitiveField());
        const Foam::vectorField aZ(mrf.DDt(probeZ)().primitiveField());

        Foam::volVectorField omega(uniformVolVector(mesh, Foam::vector::zero));
        omega.rename("MRF:omega");
        Foam::vectorField& omegaC = omega.primitiveFieldRef();
        forAll(omegaC, celli)
        {
            omegaC[celli] = Foam::vector(aY[celli].z(), aZ[celli].x(), aX[celli].y());
        }
        return toNeoNVolume(rt, omega);
    }

    // makeRelative on a zero flux leaves -(Omega x r).Sf on every face it
    // subtracts from, and 0 on the included patch faces it assigns.
    static fvcc::SurfaceField<scalar>
    buildFrameFlux(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        Foam::surfaceScalarField probe(uniformSurfaceScalar(rt.mesh, 0.0));
        mrf.makeRelative(probe);
        probe.negate();
        probe.rename("MRF:frameFlux");
        return toNeoNSurface(rt, probe);
    }

    // The difference of the zero and unit probes isolates the assignment: 0 on
    // the included patch faces makeRelative sets to zero, 1 everywhere else.
    static fvcc::SurfaceField<scalar>
    buildRelativeKeep(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        Foam::surfaceScalarField zeroProbe(uniformSurfaceScalar(rt.mesh, 0.0));
        Foam::surfaceScalarField unitProbe(uniformSurfaceScalar(rt.mesh, 1.0));
        mrf.makeRelative(zeroProbe);
        mrf.makeRelative(unitProbe);
        unitProbe -= zeroProbe;
        unitProbe.rename("MRF:relativeKeep");
        return toNeoNSurface(rt, unitProbe);
    }

    static fvcc::SurfaceField<scalar>
    buildZeroFilterMask(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        Foam::surfaceScalarField mask(mrf.zeroFilter(
            Foam::tmp<Foam::surfaceScalarField>::New(uniformSurfaceScalar(rt.mesh, 1.0))
        ));
        mask.rename("MRF:zeroFilterMask");
        return toNeoNSurface(rt, mask);
    }

    static fvcc::VolumeField<Vec3>
    buildBoundaryVelocity(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        Foam::volVectorField probe(uniformVolVector(rt.mesh, Foam::vector::zero));
        mrf.correctBoundaryVelocity(probe);
        probe.rename("MRF:boundaryVelocity");
        return toNeoNVolume(rt, probe);
    }

    // As for relativeKeep: correctBoundaryVelocity *assigns* Omega x r, so the
    // faces it touches are exactly those where the two probes agree.
    static fvcc::VolumeField<scalar>
    buildBoundaryKeep(const nf::RunTime& rt, const Foam::MRFZoneList& mrf)
    {
        const Foam::fvMesh& mesh = rt.mesh;
        Foam::volVectorField zeroProbe(uniformVolVector(mesh, Foam::vector::zero));
        Foam::volVectorField unitProbe(uniformVolVector(mesh, Foam::vector::zero));
        forAll(unitProbe.boundaryFieldRef(), patchi)
        {
            unitProbe.boundaryFieldRef()[patchi] == Foam::vector(1, 1, 1);
        }
        mrf.correctBoundaryVelocity(zeroProbe);
        mrf.correctBoundaryVelocity(unitProbe);

        Foam::volScalarField keep(
            Foam::IOobject(
                "MRF:boundaryKeep",
                mesh.time().timeName(),
                mesh,
                Foam::IOobjectOption(
                    Foam::IOobjectOption::NO_READ,
                    Foam::IOobjectOption::NO_WRITE,
                    false
                )
            ),
            mesh,
            Foam::dimensionedScalar(Foam::dimless, 1.0)
        );
        forAll(keep.boundaryFieldRef(), patchi)
        {
            Foam::scalarField kp(keep.boundaryField()[patchi].size());
            const Foam::vectorField& z = zeroProbe.boundaryField()[patchi];
            const Foam::vectorField& u = unitProbe.boundaryField()[patchi];
            forAll(kp, i)
            {
                kp[i] = (u[i] - z[i]).x();
            }
            keep.boundaryFieldRef()[patchi] == kp;
        }
        return toNeoNVolume(rt, keep);
    }

    Foam::IOMRFZoneList mrf_;
    fvcc::VolumeField<Vec3> omega_;
    fvcc::SurfaceField<scalar> frameFlux_;
    fvcc::SurfaceField<scalar> relativeKeep_;
    fvcc::SurfaceField<scalar> zeroFilterMask_;
    fvcc::VolumeField<Vec3> boundaryVelocity_;
    fvcc::VolumeField<scalar> boundaryKeep_;
    fvcc::VolumeField<Vec3> acceleration_;
};

} // namespace

namespace NeoFOAM::bindings
{

void registerMRF(nb::module_& m)
{
    nb::class_<MRFNeoN>(m, "MRFNeoN")
        // keep_alive<1,2>: the zone list and every probed field are built
        // against the runtime's OpenFOAM mesh, which must outlive them.
        .def(nb::init<nf::RunTime&>(), "runtime"_a, nb::keep_alive<1, 2>())
        .def(
            "correct_boundary_velocity",
            &MRFNeoN::correctBoundaryVelocity,
            "U"_a,
            "MRFZoneList::correctBoundaryVelocity — Omega x r on the rotating wall faces"
        )
        .def(
            "make_relative",
            &MRFNeoN::makeRelative,
            "phi"_a,
            "MRFZoneList::makeRelative — take a face flux relative to the rotating frame"
        )
        // The constants the operations are built from, so the same arithmetic can
        // be written in Python: cross(omega, U), and keep * (phi - frameFlux).
        .def_prop_ro(
            "omega",
            [](const MRFNeoN& self) -> const fvcc::VolumeField<Vec3>& { return self.omega(); },
            nb::rv_policy::reference_internal,
            "Omega per cell, zero outside the rotating zones"
        )
        .def_prop_ro(
            "frame_flux",
            [](const MRFNeoN& self) -> const fvcc::SurfaceField<scalar>&
            { return self.frameFlux(); },
            nb::rv_policy::reference_internal,
            "(Omega x r) . Sf — the rotating frame's own face flux, zero outside the zones"
        )
        .def_prop_ro(
            "relative_keep",
            [](const MRFNeoN& self) -> const fvcc::SurfaceField<scalar>&
            { return self.relativeKeep(); },
            nb::rv_policy::reference_internal,
            "0 on the rotating patch faces native assigns rather than subtracts, 1 elsewhere"
        )
        .def(
            "zero_filter",
            &MRFNeoN::zeroFilter,
            "corr"_a,
            "MRFZoneList::zeroFilter — zero an absolute-frame flux inside the MRF region"
        )
        .def(
            "acceleration",
            &MRFNeoN::acceleration,
            "U"_a,
            nb::rv_policy::reference_internal,
            "MRFZoneList::DDt(U) as a field — the frame acceleration Omega x U"
        )
        .def(
            "DDt",
            &MRFNeoN::DDt,
            "U"_a,
            nb::keep_alive<0, 1>(),
            "The momentum term + MRF.DDt(U) of UEqn.H, as an explicit source"
        );
}

} // namespace NeoFOAM::bindings
