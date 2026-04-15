// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/functionObjects/forceCoeffs.hpp"

#include "addToRunTimeSelectionTable.H"

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
    lRef_    = dict.getOrDefault<Foam::scalar>("lRef", 1.0);
    Aref_    = dict.getOrDefault<Foam::scalar>("Aref", 1.0);

    Foam::vector dragFoam = dict.getOrDefault<Foam::vector>("dragDir", Foam::vector(1, 0, 0));
    Foam::vector liftFoam = dict.getOrDefault<Foam::vector>("liftDir", Foam::vector(0, 0, 1));
    dragDir_ = NeoN::Vec3(dragFoam[0], dragFoam[1], dragFoam[2]);
    liftDir_ = NeoN::Vec3(liftFoam[0], liftFoam[1], liftFoam[2]);

    // sideDir = liftDir × dragDir  (right-hand system, mirrors OF e2 = e3 × e1)
    sideDir_ = NeoN::Vec3(
        liftDir_[1] * dragDir_[2] - liftDir_[2] * dragDir_[1],
        liftDir_[2] * dragDir_[0] - liftDir_[0] * dragDir_[2],
        liftDir_[0] * dragDir_[1] - liftDir_[1] * dragDir_[0]
    );

    return true;
}

bool ForceCoeffs::execute()
{
    if (!Forces::execute())
    {
        return false;
    }

    const ForceResult& raw = lastResult();
    const NeoN::Vec3 totalForce  = raw.pressureForce  + raw.viscousForce;
    const NeoN::Vec3 totalMoment = raw.pressureMoment + raw.viscousMoment;

    const double pDyn        = 0.5 * rhoRef_ * magUInf_ * magUInf_;
    const double forceScale  = pDyn * Aref_;
    const double momentScale = pDyn * Aref_ * lRef_;

    auto dot = [](const NeoN::Vec3& a, const NeoN::Vec3& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    Cd_      = dot(totalForce,  dragDir_) / forceScale;
    Cl_      = dot(totalForce,  liftDir_) / forceScale;
    Cs_      = dot(totalForce,  sideDir_) / forceScale;
    CmRoll_  = dot(totalMoment, dragDir_)  / momentScale;
    CmPitch_ = dot(totalMoment, sideDir_)  / momentScale;
    CmYaw_   = dot(totalMoment, liftDir_)  / momentScale;

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
            writeHeaderValue(s, "dragDir",   fmtVec3(dragDir_));
            writeHeaderValue(s, "sideDir",   fmtVec3(sideDir_));
            writeHeaderValue(s, "liftDir",   fmtVec3(liftDir_));
            writeHeaderValue(s, "rollAxis",  fmtVec3(dragDir_));
            writeHeaderValue(s, "pitchAxis", fmtVec3(sideDir_));
            writeHeaderValue(s, "yawAxis",   fmtVec3(liftDir_));
            writeHeaderValue(s, "magUInf",   fmtScalar(magUInf_));
            writeHeaderValue(s, "lRef",      fmtScalar(lRef_));
            writeHeaderValue(s, "Aref",      fmtScalar(Aref_));
            writeHeaderValue(s, "CofR",      fmtVec3(cofR_));
            writeHeader(s, "");
            writeCommented(s, "Time");
            // Alphabetical order — matches OpenFOAM's sorted coefficient map
            for (const auto* col :
                 {"Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)",
                  "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs(f)", "Cs(r)"})
            {
                writeTabbed(s, col);
            }
            s << '\n';
        }
    );

    const double CdF = 0.5 * Cd_ + CmRoll_;
    const double CdR = 0.5 * Cd_ - CmRoll_;
    const double ClF = 0.5 * Cl_ + CmPitch_;
    const double ClR = 0.5 * Cl_ - CmPitch_;
    const double CsF = 0.5 * Cs_ + CmYaw_;
    const double CsR = 0.5 * Cs_ - CmYaw_;

    writeCurrentTime(os);
    os << std::scientific << std::setprecision(writePrecision);
    for (double v : {Cd_, CdF, CdR, Cl_, ClF, ClR, CmPitch_, CmRoll_, CmYaw_, Cs_, CsF, CsR})
    {
        os << std::setw(charWidth) << v;
    }
    os << '\n';
    os.flush();
    return true;
}

} // namespace NeoFOAM
