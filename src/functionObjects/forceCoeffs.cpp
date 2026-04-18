// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/functionObjects/forceCoeffs.hpp"

#include "addToRunTimeSelectionTable.H"

using scalar = NeoN::scalar;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

const Foam::word NeoFOAM::ForceCoeffs::typeName("neoForceCoeffs");
int NeoFOAM::ForceCoeffs::debug(0);

namespace
{
Foam::functionObject::adddictionaryConstructorToTable<NeoFOAM::ForceCoeffs>
    addNeoFOAMForceCoeffsToRunTimeSelectionTable("neoForceCoeffs");
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace NeoFOAM
{

ForceCoeffs::ForceCoeffs(
    const Foam::word& name,
    const Foam::Time& runTime,
    const Foam::dictionary& dict
)
    : Forces(name, runTime, dict)
{
    read(dict);
}

bool ForceCoeffs::read(const Foam::dictionary& dict)
{
    Forces::read(dict);
    magUInf_ = dict.getOrDefault<Foam::scalar>("magUInf", 1.0);
    lRef_ = dict.getOrDefault<Foam::scalar>("lRef", 1.0);
    Aref_ = dict.getOrDefault<Foam::scalar>("Aref", 1.0);

    Foam::vector dragFoam = dict.getOrDefault<Foam::vector>("dragDir", Foam::vector(1, 0, 0));
    Foam::vector liftFoam = dict.getOrDefault<Foam::vector>("liftDir", Foam::vector(0, 0, 1));

    if (Foam::mag(dragFoam) < Foam::SMALL)
        Foam::FatalError << "neoForceCoeffs: dragDir has zero magnitude."
                         << Foam::abort(Foam::FatalError);
    if (Foam::mag(liftFoam) < Foam::SMALL)
        Foam::FatalError << "neoForceCoeffs: liftDir has zero magnitude."
                         << Foam::abort(Foam::FatalError);

    dragFoam /= Foam::mag(dragFoam);
    liftFoam /= Foam::mag(liftFoam);

    Foam::vector sideFoam = liftFoam ^ dragFoam;
    if (Foam::mag(sideFoam) < Foam::SMALL)
        Foam::FatalError << "neoForceCoeffs: dragDir and liftDir are parallel — cannot form a "
                            "right-hand coordinate system."
                         << Foam::abort(Foam::FatalError);
    sideFoam /= Foam::mag(sideFoam);

    dragDir_ = NeoN::Vec3(dragFoam[0], dragFoam[1], dragFoam[2]);
    liftDir_ = NeoN::Vec3(liftFoam[0], liftFoam[1], liftFoam[2]);
    sideDir_ = NeoN::Vec3(sideFoam[0], sideFoam[1], sideFoam[2]);

    return true;
}

bool ForceCoeffs::execute()
{
    if (!Forces::execute())
    {
        return false;
    }

    const ForceResult& raw = lastResult();
    const NeoN::Vec3 totalForce = raw.pressureForce + raw.viscousForce;
    const NeoN::Vec3 totalMoment = raw.pressureMoment + raw.viscousMoment;

    const scalar pDyn = 0.5 * rhoRef_ * magUInf_ * magUInf_;
    const scalar forceScale = pDyn * Aref_;
    const scalar momentScale = pDyn * Aref_ * lRef_;

    if (forceScale == 0 || momentScale == 0)
    {
        Foam::FatalError << "neoForceCoeffs: scaling factor is zero "
                         << "(rhoRef=" << rhoRef_ << ", magUInf=" << magUInf_ << ", Aref=" << Aref_
                         << ", lRef=" << lRef_ << "). "
                         << "Cannot compute force coefficients." << Foam::abort(Foam::FatalError);
    }

    auto dot = [](const NeoN::Vec3& a, const NeoN::Vec3& b)
    { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };

    Cd_ = dot(totalForce, dragDir_) / forceScale;
    Cl_ = dot(totalForce, liftDir_) / forceScale;
    Cs_ = dot(totalForce, sideDir_) / forceScale;
    CmRoll_ = dot(totalMoment, dragDir_) / momentScale;
    CmPitch_ = dot(totalMoment, sideDir_) / momentScale;
    CmYaw_ = dot(totalMoment, liftDir_) / momentScale;

    return true;
}

bool ForceCoeffs::write()
{
    // neoForceCoeffs writes coefficient.dat only.
    // force.dat and moment.dat are written exclusively by neoForces.
    auto& os = getOrCreateFile(
        "coefficient.dat",
        [&](std::ostream& s)
        {
            writeHeader(s, "Force and moment coefficients");
            writeHeaderValue(s, "dragDir", fmtVec3(dragDir_));
            writeHeaderValue(s, "sideDir", fmtVec3(sideDir_));
            writeHeaderValue(s, "liftDir", fmtVec3(liftDir_));
            writeHeaderValue(s, "rollAxis", fmtVec3(dragDir_));
            writeHeaderValue(s, "pitchAxis", fmtVec3(sideDir_));
            writeHeaderValue(s, "yawAxis", fmtVec3(liftDir_));
            writeHeaderValue(s, "magUInf", fmtScalar(magUInf_));
            writeHeaderValue(s, "lRef", fmtScalar(lRef_));
            writeHeaderValue(s, "Aref", fmtScalar(Aref_));
            writeHeaderValue(s, "CofR", fmtVec3(cofR_));
            writeHeader(s, "");
            writeCommented(s, "Time");
            // Alphabetical order — matches OpenFOAM's sorted coefficient map
            for (const auto* col :
                 {"Cd",
                  "Cd(f)",
                  "Cd(r)",
                  "Cl",
                  "Cl(f)",
                  "Cl(r)",
                  "CmPitch",
                  "CmRoll",
                  "CmYaw",
                  "Cs",
                  "Cs(f)",
                  "Cs(r)"})
            {
                writeTabbed(s, col);
            }
            s << '\n';
        }
    );

    const scalar cdF = 0.5 * Cd_ + CmRoll_;
    const scalar cdR = 0.5 * Cd_ - CmRoll_;
    const scalar clF = 0.5 * Cl_ + CmPitch_;
    const scalar clR = 0.5 * Cl_ - CmPitch_;
    const scalar csF = 0.5 * Cs_ + CmYaw_;
    const scalar csR = 0.5 * Cs_ - CmYaw_;

    writeCurrentTime(os);
    os << std::scientific << std::setprecision(writePrecision);
    for (scalar v : {Cd_, cdF, cdR, Cl_, clF, clR, CmPitch_, CmRoll_, CmYaw_, Cs_, csF, csR})
    {
        os << std::setw(charWidth) << v;
    }
    os << '\n';
    os.flush();
    return true;
}

} // namespace NeoFOAM
