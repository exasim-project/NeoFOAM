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
    lRef_ = dict.getOrDefault<Foam::scalar>("lRef", 1.0);
    Aref_ = dict.getOrDefault<Foam::scalar>("Aref", 1.0);
    return true;
}

NeoN::Vec3 ForceCoeffs::normalizeForce(const NeoN::Vec3& f) const
{
    const NeoN::scalar dyn = 0.5 * rhoRef_ * magUInf_ * magUInf_ * Aref_;
    return f / (dyn + NeoN::scalar {1e-300});
}

NeoN::Vec3 ForceCoeffs::normalizeMoment(const NeoN::Vec3& m) const
{
    const NeoN::scalar dyn = 0.5 * rhoRef_ * magUInf_ * magUInf_ * Aref_ * lRef_;
    return m / (dyn + NeoN::scalar {1e-300});
}

bool ForceCoeffs::execute()
{
    if (!Forces::execute())
    {
        return false;
    }

    const ForceResult& raw = lastResult();
    coeffResult_.pressureForce = normalizeForce(raw.pressureForce);
    coeffResult_.viscousForce = normalizeForce(raw.viscousForce);
    coeffResult_.pressureMoment = normalizeMoment(raw.pressureMoment);
    coeffResult_.viscousMoment = normalizeMoment(raw.viscousMoment);

    return true;
}

bool ForceCoeffs::write()
{
    // Also write raw forces via base class
    Forces::write();

    auto& os = getOrCreateFile(
        "coefficient.dat",
        [&](std::ostream& s)
        {
            writeHeader(s, "Force and moment coefficients");
            writeHeaderValue(s, "magUInf", fmtScalar(magUInf_));
            writeHeaderValue(s, "lRef", fmtScalar(lRef_));
            writeHeaderValue(s, "Aref", fmtScalar(Aref_));
            writeHeaderValue(s, "CofR", fmtVec3(cofR_));
            writeHeader(s, "");
            writeCommented(s, "Time");
            for (const auto* col : {"Cx", "Cy", "Cz", "CmX", "CmY", "CmZ"})
            {
                writeTabbed(s, col);
            }
            s << '\n';
        }
    );

    writeCurrentTime(os);
    os << std::scientific << std::setprecision(writePrecision);
    for (int i = 0; i < 3; ++i)
    {
        os << std::setw(charWidth) << coeffResult_.pressureForce[i];
    }
    for (int i = 0; i < 3; ++i)
    {
        os << std::setw(charWidth) << coeffResult_.pressureMoment[i];
    }
    os << '\n';
    os.flush();
    return true;
}

} // namespace NeoFOAM
