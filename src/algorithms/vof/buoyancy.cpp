// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>
#include <string>

#include "NeoN/NeoN.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/algorithms/vof/buoyancy.hpp"
#include "NeoFOAM/auxiliary/readers.hpp"
#include "detail.hpp"
#include "foamGeometry.hpp"

#include "surfaceFields.H"
#include "uniformDimensionedFields.H"
#include "volFields.H"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

namespace
{

using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vec3;

// OpenFOAM doubleScalar constants (doubleScalar.H).
constexpr scalar small = 1.0e-15;
constexpr scalar rootVSmall = 1.0e-150;

void requireUncoupled(const NeoN::UnstructuredMesh& mesh, const std::string& what)
{
    if (mesh.nProcBoundaryFaces() > 0)
    {
        throw std::runtime_error(what + ": coupled (processor) patches are not supported");
    }
}

// gh.H's ghRef = g & (cmptMag(g)/|g|)*hRef, with readhRef.H's default hRef = 0.
Foam::dimensionedScalar readGhRef(RunTime& rt, const Foam::uniformDimensionedVectorField& g)
{
    const Foam::uniformDimensionedScalarField hRef(
        Foam::IOobject(
            "hRef",
            rt.mesh.time().constant(),
            rt.mesh.thisDb(),
            Foam::IOobject::READ_IF_PRESENT,
            Foam::IOobject::NO_WRITE,
            Foam::IOobject::NO_REGISTER
        ),
        Foam::dimensionedScalar(Foam::word::null, Foam::dimLength, Foam::Zero)
    );
    return Foam::mag(g.value()) > small
             ? g & (Foam::cmptMag(g.value()) / Foam::mag(g.value())) * hRef
             : Foam::dimensionedScalar("ghRef", g.dimensions() * Foam::dimLength, 0);
}

Foam::uniformDimensionedVectorField readGravity(RunTime& rt)
{
    return Foam::uniformDimensionedVectorField(Foam::IOobject(
        "g",
        rt.mesh.time().constant(),
        rt.mesh.thisDb(),
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE,
        Foam::IOobject::NO_REGISTER
    ));
}

// Whether the snGradSchemes entry of psi asks for the non-orthogonal correction; only the two
// schemes interFoam cases use are implemented.
bool snGradCorrected(RunTime& rt, const std::string& name)
{
    const Foam::ITstream& scheme = rt.mesh.snGradScheme("snGrad(" + name + ")");
    if (scheme.size() == 1 && scheme[0].isWord())
    {
        if (scheme[0].wordToken() == "corrected")
        {
            return true;
        }
        if (scheme[0].wordToken() == "uncorrected")
        {
            return false;
        }
    }
    throw std::runtime_error(
        "snGrad(" + name + "): only the corrected and uncorrected snGrad schemes are supported"
    );
}

void requireGaussLinearGrad(RunTime& rt, const std::string& name)
{
    const Foam::ITstream& scheme = rt.mesh.gradScheme("grad(" + name + ")");
    const bool gaussLinear = scheme.size() == 2 && scheme[0].isWord() && scheme[1].isWord()
                          && scheme[0].wordToken() == "Gauss" && scheme[1].wordToken() == "linear";
    if (!gaussLinear)
    {
        throw std::runtime_error(
            "snGrad(" + name + ") corrected: the correction needs grad(" + name
            + ") under Gauss linear"
        );
    }
}

// fvc::snGrad(psi) into out (internal and boundary faces): snGradScheme::snGrad with the
// scheme's nonOrthDeltaCoeffs, plus correctedSnGrad's fullGradCorrection on the internal faces
// (non-coupled boundary correction vectors are zero). Boundary: fvPatchField::snGrad, which
// fixedGradient patches (fixedFluxPressure among them) override to return the gradient.
void snGrad(RunTime& rt, const fvcc::VolumeField<scalar>& psi, fvcc::SurfaceField<scalar>& out)
{
    const auto& mesh = psi.mesh();
    requireUncoupled(mesh, "snGrad(" + psi.name + ")");
    const auto exec = psi.exec();
    const FoamGeometry& geometry = foamGeometry(rt);
    const auto [own, nei, deltaCoeffs, psiI] = NeoN::views(
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        geometry.nonOrthDeltaCoeffs,
        psi.internalVector()
    );
    auto sn = out.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { sn[f] = deltaCoeffs[f] * (psiI[nei[f]] - psiI[own[f]]); },
        "buoyancy::snGradInternal"
    );

    if (snGradCorrected(rt, psi.name))
    {
        requireGaussLinearGrad(rt, psi.name);
        const auto weights = geometry.weights.view();
        const auto corrVecs = geometry.nonOrthCorrectionVectors.view();
        NeoN::Vector<Vec3> gradVec = gaussLinearGrad(psi, weights);
        const auto g = gradVec.view();
        NeoN::parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx f) {
                const scalar w = weights[f];
                const Vec3 gP = g[own[f]];
                const Vec3 gN = g[nei[f]];
                const Vec3 gf(
                    faceValue(w, gP[0], gN[0]),
                    faceValue(w, gP[1], gN[1]),
                    faceValue(w, gP[2], gN[2])
                );
                sn[f] += corrVecs[f] & gf;
            },
            "buoyancy::snGradCorrection"
        );
    }

    const auto [bOwn, bDeltaCoeffs, psiB, refGrad] = NeoN::views(
        mesh.boundaryMesh().faceOwners(),
        mesh.boundaryMesh().deltaCoeffs(),
        psi.boundaryData().value(),
        psi.boundaryData().refGrad()
    );
    auto snB = out.boundaryData().value().view();
    const auto bcs = psi.boundaryConditions();
    for (std::size_t patchi = 0; patchi < bcs.size(); ++patchi)
    {
        const std::string type = bcs[patchi].name();
        const bool gradientPatch = type == "fixedGradient" || type == "fixedFluxPressure";
        NeoN::parallelFor(
            exec,
            psi.boundaryData().range(static_cast<localIdx>(patchi)),
            NEON_LAMBDA(const localIdx bf) {
                snB[bf] =
                    gradientPatch ? refGrad[bf] : bDeltaCoeffs[bf] * (psiB[bf] - psiI[bOwn[bf]]);
            },
            "buoyancy::snGradBoundary"
        );
    }
}

// fvc::reconstruct's per-cell solve: Tensor::safeInv(T) & b, as TensorI.H evaluates it.
struct Tensor9
{
    scalar xx, xy, xz, yx, yy, yz, zx, zy, zz;
};

KOKKOS_INLINE_FUNCTION scalar det(const Tensor9& t)
{
    return t.xx * t.yy * t.zz + t.xy * t.yz * t.zx + t.xz * t.yx * t.zy - t.xx * t.yz * t.zy
         - t.xy * t.yx * t.zz - t.xz * t.yy * t.zx;
}

// t.adjunct()/detval
KOKKOS_INLINE_FUNCTION Tensor9 adjunctOver(const Tensor9& t, scalar detval)
{
    return {
        (t.yy * t.zz - t.zy * t.yz) / detval,
        (t.xz * t.zy - t.xy * t.zz) / detval,
        (t.xy * t.yz - t.xz * t.yy) / detval,
        (t.zx * t.yz - t.yx * t.zz) / detval,
        (t.xx * t.zz - t.xz * t.zx) / detval,
        (t.yx * t.xz - t.xx * t.yz) / detval,
        (t.yx * t.zy - t.yy * t.zx) / detval,
        (t.xy * t.zx - t.xx * t.zy) / detval,
        (t.xx * t.yy - t.yx * t.xy) / detval
    };
}

KOKKOS_INLINE_FUNCTION Tensor9 safeInv(Tensor9 t)
{
    const scalar magSqrXx = t.xx * t.xx;
    const scalar magSqrYy = t.yy * t.yy;
    const scalar magSqrZz = t.zz * t.zz;
    const scalar threshold = small * (magSqrXx + magSqrYy + magSqrZz);
    const bool smallXx = magSqrXx < threshold;
    const bool smallYy = magSqrYy < threshold;
    const bool smallZz = magSqrZz < threshold;
    // 2-D meshes: a unit diagonal in the empty direction makes the tensor invertible.
    if (smallXx) t.xx += 1.0;
    if (smallYy) t.yy += 1.0;
    if (smallZz) t.zz += 1.0;
    const scalar detval = det(t);
    if (Kokkos::fabs(detval) < rootVSmall)
    {
        return {0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    Tensor9 inv = adjunctOver(t, detval);
    if (smallXx) inv.xx -= 1.0;
    if (smallYy) inv.yy -= 1.0;
    if (smallZz) inv.zz -= 1.0;
    return inv;
}

// The face flux of a surface field, f < nInternal an internal face, else boundary face
// f - nInternal (the indexing of cellFaces).
struct FieldFlux
{
    NeoN::View<const scalar> internal;
    NeoN::View<const scalar> boundary;
    localIdx nInternal;

    KOKKOS_INLINE_FUNCTION scalar operator()(localIdx f) const
    {
        return f < nInternal ? internal[f] : boundary[f - nInternal];
    }
};

// pEqn.H's reconstruct argument (phig - pFlux)/rAUf on one face (indexing as FieldFlux).
struct BuoyantFlux
{
    FieldFlux phig;
    FieldFlux pFlux;
    FieldFlux rAUf;

    KOKKOS_INLINE_FUNCTION scalar operator()(localIdx f) const
    {
        return (phig(f) - pFlux(f)) / rAUf(f);
    }
};

struct StoreVector
{
    NeoN::View<Vec3> out;

    KOKKOS_INLINE_FUNCTION void operator()(localIdx c, const Vec3& value) const { out[c] = value; }
};

// U = HbyA + rAU*value
struct StoreCorrectedVelocity
{
    NeoN::View<const Vec3> hByA;
    NeoN::View<const scalar> rAU;
    NeoN::View<Vec3> U;

    KOKKOS_INLINE_FUNCTION void operator()(localIdx c, const Vec3& value) const
    {
        const scalar r = rAU[c];
        const Vec3 h = hByA[c];
        U[c] = Vec3(h[0] + r * value[0], h[1] + r * value[1], h[2] + r * value[2]);
    }
};

FieldFlux fieldFlux(const fvcc::SurfaceField<scalar>& ssf)
{
    return {
        ssf.internalVector().view(),
        ssf.boundaryData().value().view(),
        ssf.mesh().nInternalFaces()
    };
}

// fvc::reconstruct of faceFlux, handed per cell to store(cell, value). Each cell sums SfHat*Sf
// and SfHat*flux over its faces in fvc::surfaceSum's order.
template<typename FaceFlux, typename Store>
void reconstructCells(const NeoN::UnstructuredMesh& mesh, FaceFlux faceFlux, Store store)
{
    requireUncoupled(mesh, "reconstruct");
    const CellFaces& map = cellFaces(mesh);
    const auto [offsets, faces, sf, magSf, bSf, bMagSf] = NeoN::views(
        map.offsets,
        map.faces,
        mesh.faceNormals(),
        mesh.faceAreas(),
        mesh.boundaryMesh().faceNormals(),
        mesh.boundaryMesh().faceAreas()
    );
    const localIdx nInternal = mesh.nInternalFaces();
    NeoN::parallelFor(
        mesh.exec(),
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx c) {
            Tensor9 t {0, 0, 0, 0, 0, 0, 0, 0, 0};
            scalar bx = 0, by = 0, bz = 0;
            for (localIdx i = offsets[c]; i < offsets[c + 1]; ++i)
            {
                const localIdx f = faces[i];
                const bool internal = f < nInternal;
                const Vec3 s = internal ? sf[f] : bSf[f - nInternal];
                const scalar mag = internal ? magSf[f] : bMagSf[f - nInternal];
                // SfHat = Sf/magSf, then the outer product SfHat*Sf and SfHat*flux
                const scalar hx = s[0] / mag;
                const scalar hy = s[1] / mag;
                const scalar hz = s[2] / mag;
                const scalar flux = faceFlux(f);
                t.xx += hx * s[0];
                t.xy += hx * s[1];
                t.xz += hx * s[2];
                t.yx += hy * s[0];
                t.yy += hy * s[1];
                t.yz += hy * s[2];
                t.zx += hz * s[0];
                t.zy += hz * s[1];
                t.zz += hz * s[2];
                bx += hx * flux;
                by += hy * flux;
                bz += hz * flux;
            }
            const Tensor9 inv = safeInv(t);
            store(
                c,
                Vec3(
                    inv.xx * bx + inv.xy * by + inv.xz * bz,
                    inv.yx * bx + inv.yy * by + inv.yz * bz,
                    inv.zx * bx + inv.zy * by + inv.zz * bz
                )
            );
        },
        "buoyancy::reconstruct"
    );
}

} // namespace

fvcc::VolumeField<scalar> createGh(RunTime& rt)
{
    const Foam::uniformDimensionedVectorField g = readGravity(rt);
    const Foam::volScalarField gh("gh", (g & rt.mesh.C()) - readGhRef(rt, g));
    return constructFrom(rt.exec, rt.nfMesh, gh);
}

fvcc::SurfaceField<scalar> createGhf(RunTime& rt)
{
    const Foam::uniformDimensionedVectorField g = readGravity(rt);
    const Foam::surfaceScalarField ghf("ghf", (g & rt.mesh.Cf()) - readGhRef(rt, g));
    return constructFrom(rt.exec, rt.nfMesh, ghf);
}

void gravityFaceForce(
    RunTime& rt,
    const fvcc::SurfaceField<scalar>& ghf,
    const fvcc::VolumeField<scalar>& rho,
    fvcc::SurfaceField<scalar>& force
)
{
    snGrad(rt, rho, force);
    const auto [ghfI, ghfB] = NeoN::views(ghf.internalVector(), ghf.boundaryData().value());
    auto [fI, fB] = NeoN::views(force.internalVector(), force.boundaryData().value());
    NeoN::parallelFor(
        rho.exec(),
        {0, fI.size()},
        NEON_LAMBDA(const localIdx f) { fI[f] = -(ghfI[f] * fI[f]); },
        "buoyancy::gravityForceInternal"
    );
    NeoN::parallelFor(
        rho.exec(),
        {0, fB.size()},
        NEON_LAMBDA(const localIdx bf) { fB[bf] = -(ghfB[bf] * fB[bf]); },
        "buoyancy::gravityForceBoundary"
    );
}

void buoyancyFlux(
    const fvcc::SurfaceField<scalar>& force,
    const fvcc::SurfaceField<scalar>& rAUf,
    fvcc::SurfaceField<scalar>& phig
)
{
    const auto& mesh = force.mesh();
    const auto [fI, rI, magSf] =
        NeoN::views(force.internalVector(), rAUf.internalVector(), mesh.faceAreas());
    auto pI = phig.internalVector().view();
    NeoN::parallelFor(
        force.exec(),
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { pI[f] = fI[f] * rI[f] * magSf[f]; },
        "buoyancy::fluxInternal"
    );
    const auto [fB, rB, bMagSf] = NeoN::views(
        force.boundaryData().value(),
        rAUf.boundaryData().value(),
        mesh.boundaryMesh().faceAreas()
    );
    auto pB = phig.boundaryData().value().view();
    NeoN::parallelFor(
        force.exec(),
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { pB[bf] = fB[bf] * rB[bf] * bMagSf[bf]; },
        "buoyancy::fluxBoundary"
    );
}

void buoyantMomentumFlux(
    RunTime& rt,
    const fvcc::SurfaceField<scalar>& force,
    const fvcc::VolumeField<scalar>& p_rgh,
    fvcc::SurfaceField<scalar>& flux
)
{
    snGrad(rt, p_rgh, flux);
    const auto& mesh = force.mesh();
    const auto [fI, magSf] = NeoN::views(force.internalVector(), mesh.faceAreas());
    auto oI = flux.internalVector().view();
    NeoN::parallelFor(
        force.exec(),
        {0, mesh.nInternalFaces()},
        NEON_LAMBDA(const localIdx f) { oI[f] = (fI[f] - oI[f]) * magSf[f]; },
        "buoyancy::momentumFluxInternal"
    );
    const auto [fB, bMagSf] =
        NeoN::views(force.boundaryData().value(), mesh.boundaryMesh().faceAreas());
    auto oB = flux.boundaryData().value().view();
    NeoN::parallelFor(
        force.exec(),
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { oB[bf] = (fB[bf] - oB[bf]) * bMagSf[bf]; },
        "buoyancy::momentumFluxBoundary"
    );
}

void reconstruct(const fvcc::SurfaceField<scalar>& ssf, fvcc::VolumeField<Vec3>& out)
{
    const auto& mesh = ssf.mesh();
    reconstructCells(mesh, fieldFlux(ssf), StoreVector {out.internalVector().view()});
    // extrapolatedCalculated: every boundary face takes its owner-cell value.
    const auto [bOwn, oI] = NeoN::views(mesh.boundaryMesh().faceOwners(), out.internalVector());
    auto oB = out.boundaryData().value().view();
    NeoN::parallelFor(
        out.exec(),
        {0, mesh.nBoundaryFaces()},
        NEON_LAMBDA(const localIdx bf) { oB[bf] = oI[bOwn[bf]]; },
        "buoyancy::reconstructBoundary"
    );
}

void updateVelocityBuoyant(
    const fvcc::VolumeField<Vec3>& hByA,
    const fvcc::VolumeField<scalar>& rAU,
    const fvcc::SurfaceField<scalar>& phig,
    const fvcc::SurfaceField<scalar>& pFlux,
    const fvcc::SurfaceField<scalar>& rAUf,
    fvcc::VolumeField<Vec3>& U
)
{
    reconstructCells(
        U.mesh(),
        BuoyantFlux {fieldFlux(phig), fieldFlux(pFlux), fieldFlux(rAUf)},
        StoreCorrectedVelocity {
            hByA.internalVector().view(),
            rAU.internalVector().view(),
            U.internalVector().view()
        }
    );
}

void updateStaticPressure(
    const fvcc::VolumeField<scalar>& p_rgh,
    const fvcc::VolumeField<scalar>& rho,
    const fvcc::VolumeField<scalar>& gh,
    fvcc::VolumeField<scalar>& p
)
{
    const auto [prI, rI, gI] =
        NeoN::views(p_rgh.internalVector(), rho.internalVector(), gh.internalVector());
    auto pI = p.internalVector().view();
    NeoN::parallelFor(
        p.exec(),
        {0, pI.size()},
        NEON_LAMBDA(const localIdx c) { pI[c] = prI[c] + rI[c] * gI[c]; },
        "buoyancy::staticPressureInternal"
    );
    const auto [prB, rB, gB] = NeoN::views(
        p_rgh.boundaryData().value(),
        rho.boundaryData().value(),
        gh.boundaryData().value()
    );
    auto pB = p.boundaryData().value().view();
    NeoN::parallelFor(
        p.exec(),
        {0, pB.size()},
        NEON_LAMBDA(const localIdx bf) { pB[bf] = prB[bf] + rB[bf] * gB[bf]; },
        "buoyancy::staticPressureBoundary"
    );
}

} // namespace NeoFOAM
