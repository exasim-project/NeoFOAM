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
        "forceCoeffs.dat",
        "# Time\tCd.x\tCd.y\tCd.z\tCm.x\tCm.y\tCm.z"
    );

    os << time_.value()
       << "\t" << coeffResult_.pressureForce[0]
       << "\t" << coeffResult_.pressureForce[1]
       << "\t" << coeffResult_.pressureForce[2]
       << "\t" << coeffResult_.pressureMoment[0]
       << "\t" << coeffResult_.pressureMoment[1]
       << "\t" << coeffResult_.pressureMoment[2]
       << "\n";

    os.flush();
    return true;
}

} // namespace NeoFOAM
