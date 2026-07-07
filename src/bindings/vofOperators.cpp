// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <nanobind/nanobind.h>
#include <string>

#include "NeoN/NeoN.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/uncorrected.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/surfaceIntegrate.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "fvCFD.H"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

namespace NeoFOAM::bindings
{

namespace
{

// Interface unit-normal flux nHatf = nHatfv & Sf, with
//   nHatfv     = gradAlphaf / (|gradAlphaf| + deltaN),
//   gradAlphaf = interpolate(grad(alpha1))   (Gauss-Green grad, linear face interp),
//   deltaN     = 1e-8 / cbrt(mean(cell volume)).
// Faithful to interfaceProperties::calculateK (interfaceProperties.C:108-166) minus the
// contact-angle correction and curvature smoothing. Scatters internal + boundary faces via
// Kokkos parallelFor over the mesh Sf views (reconstruct.cpp pattern). Factored out so
// surface_tension_force reuses the identical nHatf (single source of truth).
fvcc::SurfaceField<NeoN::scalar> computeNHatf(const fvcc::VolumeField<NeoN::scalar>& alpha1)
{
    const auto exec = alpha1.exec();
    const auto& mesh = alpha1.mesh();

    // Cell gradient of alpha (Gauss-Green == OpenFOAM "nHat" Gauss linear default).
    fvcc::GaussGreenGrad gg(exec, mesh);
    fvcc::VolumeField<NeoN::Vec3> gradAlpha = gg.grad(alpha1);

    // Interpolated face gradient gradAlphaf = interpolate(gradAlpha) (linear).
    fvcc::SurfaceInterpolation<NeoN::Vec3> interp(
        exec, mesh, NeoN::Input {NeoN::TokenList {std::string("linear")}}
    );
    fvcc::SurfaceField<NeoN::Vec3> gradAlphaf = interp.interpolate(gradAlpha);

    // deltaN = 1e-8 / cbrt(mean(cell volume)); must match interfaceProperties.C:194 exactly.
    const auto nCells = mesh.nCells();
    auto vHost = mesh.cellVolumes().copyToHost();
    auto vView = vHost.view();
    double sumV = 0.0;
    for (NeoN::localIdx c = 0; c < nCells; ++c) sumV += vView[c];
    const NeoN::scalar deltaN = 1e-8 / std::cbrt(sumV / static_cast<double>(nCells));

    fvcc::SurfaceField<NeoN::scalar> nHatf(
        exec, "nHatf", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
    );

    const auto Sf = mesh.faceNormals().view();
    const auto gaf = gradAlphaf.internalVector().view();
    auto nh = nHatf.internalVector().view();
    NeoN::parallelFor(
        exec, {0, mesh.nInternalFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const NeoN::Vec3 g = gaf[f];
            const NeoN::scalar mg = NeoN::mag(g) + deltaN;
            nh[f] = (g / mg) & Sf[f];
        },
        "vof::nHatfInternal"
    );

    const auto bSf = mesh.boundaryMesh().faceNormals().view();
    const auto bgaf = gradAlphaf.boundaryData().value().view();
    auto bnh = nHatf.boundaryData().value().view();
    NeoN::parallelFor(
        exec, {0, mesh.nBoundaryFaces()},
        KOKKOS_LAMBDA(const NeoN::localIdx bf) {
            const NeoN::Vec3 g = bgaf[bf];
            const NeoN::scalar mg = NeoN::mag(g) + deltaN;
            bnh[bf] = (g / mg) & bSf[bf];
        },
        "vof::nHatfBoundary"
    );

    return nHatf;
}

} // namespace

void registerVofOperators(nb::module_& m)
{
    // snGrad(field): face-normal gradient (uncorrected scheme). Returns a fresh
    // SurfaceField<scalar> by value.
    m.def(
        "sn_grad",
        [](const fvcc::VolumeField<NeoN::scalar>& f) -> fvcc::SurfaceField<NeoN::scalar>
        {
            fvcc::FaceNormalGradient<NeoN::scalar> fng(
                f.exec(), f.mesh(),
                NeoN::Input {NeoN::TokenList {std::string("uncorrected")}}
            );
            return fng.faceNormalGrad(f);
        },
        "field"_a,
        "Face-normal gradient snGrad(field) (uncorrected)"
    );

    // Element-wise product of two scalar volume fields (internal + boundary), as a
    // fresh unregistered VolumeField. Used to form rho*rAU for interFoam's ddtCorr
    // weighting interpolate(rho*rAU) (NeoN has no VolumeField __mul__ binding).
    m.def(
        "mul_scalar_volume",
        [](const fvcc::VolumeField<NeoN::scalar>& a, const fvcc::VolumeField<NeoN::scalar>& b)
            -> fvcc::VolumeField<NeoN::scalar>
        {
            fvcc::VolumeField<NeoN::scalar> res(
                a.exec(), a.name + "*" + b.name, a.mesh(),
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(a.mesh())
            );
            res.internalVector() = a.internalVector();
            NeoN::mul(res.internalVector(), b.internalVector());
            res.boundaryData().value() = a.boundaryData().value();
            NeoN::mul(res.boundaryData().value(), b.boundaryData().value());
            return res;
        },
        "a"_a,
        "b"_a,
        "Element-wise product of two scalar volume fields (internal + boundary)"
    );

    // |Sf|: face-area magnitudes as a surface field, built from the OpenFOAM mesh's
    // magSf() and converted (unregistered) — mirrors the ghf/rhoPhi conversion path.
    m.def(
        "mag_sf",
        [](nf::RunTime& rt) -> fvcc::SurfaceField<NeoN::scalar>
        {
            Foam::surfaceScalarField magSf("magSf", rt.mesh.magSf());
            return nf::constructFrom(rt.exec, rt.nfMesh, magSf);
        },
        "runtime"_a,
        "Face-area magnitudes |Sf| as a surface field"
    );

    // Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK).
    m.def(
        "interface_nhatf",
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1)
            -> fvcc::SurfaceField<NeoN::scalar> { return computeNHatf(alpha1); },
        "runtime"_a,
        "alpha1"_a,
        "Interface unit-normal flux nHatf = nHatfv & Sf (interfaceProperties::calculateK)"
    );

    // Surface-tension face force fSigma = interpolate(sigma*K)*snGrad(alpha1), with the
    // curvature K = -div(nHatf) = -surfaceIntegrate(nHatf). Mirrors
    // interfaceProperties::surfaceTensionForce (interfaceProperties.C:237-241).
    m.def(
        "surface_tension_force",
        [](nf::RunTime&, const fvcc::VolumeField<NeoN::scalar>& alpha1, double sigma)
            -> fvcc::SurfaceField<NeoN::scalar>
        {
            const auto exec = alpha1.exec();
            const auto& mesh = alpha1.mesh();

            fvcc::SurfaceField<NeoN::scalar> nHatf = computeNHatf(alpha1);

            // sigmaK = sigma*K = -sigma*surfaceIntegrate(nHatf). surfaceIntegrate divides by
            // V and scales by the operator coefficient, so passing -sigma yields sigma*K in
            // one pass (verified against reconstruct.cpp / mules.cpp surfaceIntegrate usage).
            fvcc::VolumeField<NeoN::scalar> sigmaK(
                exec, "sigmaK", mesh,
                fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh)
            );
            NeoN::fill(sigmaK.internalVector(), 0.0);
            fvcc::surfaceIntegrate<NeoN::scalar>(
                exec,
                mesh.nInternalFaces(),
                mesh.faceNeighbors().view(),
                mesh.faceOwners().view(),
                mesh.boundaryMesh().faceOwners().view(),
                nHatf.internalVector().view(),
                nHatf.boundaryData().value().view(),
                mesh.cellVolumes().view(),
                sigmaK.internalVector().view(),
                NeoN::dsl::Coeff(-sigma)
            );
            sigmaK.correctBoundaryConditions();

            // sigmaKf = interpolate(sigmaK) (linear).
            fvcc::SurfaceInterpolation<NeoN::scalar> interpS(
                exec, mesh, NeoN::Input {NeoN::TokenList {std::string("linear")}}
            );
            fvcc::SurfaceField<NeoN::scalar> sigmaKf = interpS.interpolate(sigmaK);

            // snGrad(alpha1) (uncorrected == corrected on the orthogonal damBreak mesh).
            fvcc::FaceNormalGradient<NeoN::scalar> fng(
                exec, mesh, NeoN::Input {NeoN::TokenList {std::string("uncorrected")}}
            );
            fvcc::SurfaceField<NeoN::scalar> snAlpha = fng.faceNormalGrad(alpha1);

            // fSigma = sigmaKf * snGrad(alpha1) (element-wise internal + boundary).
            fvcc::SurfaceField<NeoN::scalar> fSigma(
                exec, "fSigma", mesh,
                fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
            );
            fSigma.internalVector() = sigmaKf.internalVector();
            NeoN::mul(fSigma.internalVector(), snAlpha.internalVector());
            fSigma.boundaryData().value() = sigmaKf.boundaryData().value();
            NeoN::mul(fSigma.boundaryData().value(), snAlpha.boundaryData().value());
            return fSigma;
        },
        "runtime"_a,
        "alpha1"_a,
        "sigma"_a,
        "Surface-tension face force sigma*K*snGrad(alpha1) (interfaceProperties)"
    );
}
} // namespace NeoFOAM::bindings
