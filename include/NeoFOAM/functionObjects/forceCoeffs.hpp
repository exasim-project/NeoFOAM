// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoFOAM/functionObjects/forces.hpp"

namespace NeoFOAM
{

/**
 * @class ForceCoeffs
 * @brief Normalised aerodynamic force coefficients derived from GPU force integrals.
 *
 * Extends Forces by projecting raw forces/moments onto user-defined drag, lift and
 * side direction vectors and dividing by the dynamic pressure reference quantities to
 * produce dimensionless coefficients.  The output exactly matches OpenFOAM's built-in
 * forceCoeffs function object.
 *
 * Registered as @c neoForceCoeffs in OpenFOAM's runtime selection table:
 * @code
 * functions
 * {
 *     aeroCoeffs
 *     {
 *         type      neoForceCoeffs;
 *         libs      (NeoFOAM);
 *         patches   (wing);
 *         pName     p;
 *         rhoInf    1.225;
 *         pRef      0.0;
 *         magUInf   30.0;
 *         lRef      1.0;
 *         Aref      0.5;
 *         CofR      (0 0 0);
 *         dragDir   (1 0 0);
 *         liftDir   (0 0 1);
 *     }
 * }
 * @endcode
 *
 * @par Output file: coefficient.dat
 * 12 scalar columns in alphabetical order (matches OpenFOAM):
 * @code
 * # Time  Cd  Cd(f)  Cd(r)  Cl  Cl(f)  Cl(r)  CmPitch  CmRoll  CmYaw  Cs  Cs(f)  Cs(r)
 * @endcode
 *
 * The front/rear axle split follows OpenFOAM's convention:
 *   Cd(f) = 0.5*Cd + CmRoll;   Cd(r) = 0.5*Cd - CmRoll
 *   Cl(f) = 0.5*Cl + CmPitch;  Cl(r) = 0.5*Cl - CmPitch
 *   Cs(f) = 0.5*Cs + CmYaw;    Cs(r) = 0.5*Cs - CmYaw
 *
 * @note Viscous force contributions are zero in v1.
 * @note MPI parallel execution is not yet supported.
 * @note write() produces coefficient.dat only — force.dat and moment.dat are written
 *       exclusively by the neoForces function object.
 */
class ForceCoeffs : public Forces
{
public:

    // OpenFOAM RTST members (manual definition — see forces.hpp comment).
    static const Foam::word typeName;
    static int debug;
    virtual const Foam::word& type() const override { return typeName; }

    //- Construct from name, Time and dictionary (RTST constructor signature)
    ForceCoeffs(const Foam::word& name, const Foam::Time& runTime, const Foam::dictionary& dict);

    //- Destructor
    virtual ~ForceCoeffs() = default;

    virtual bool read(const Foam::dictionary& dict) override;

    /**
     * @brief Run Forces::execute() then project forces/moments onto direction vectors.
     *
     * Computes Cd, Cl, Cs, CmRoll, CmPitch, CmYaw as scalars ready for write().
     */
    virtual bool execute() override;

    /**
     * @brief Write normalised coefficients to coefficient.dat.
     *
     * Does NOT write force.dat or moment.dat — those are only written by neoForces.
     */
    virtual bool write() override;

private:

    NeoN::scalar magUInf_ {1.0};
    NeoN::scalar lRef_ {1.0};
    NeoN::scalar Aref_ {1.0};

    // Direction vectors in global coordinates.
    // dragDir = e1, liftDir = e3, sideDir = liftDir × dragDir (right-hand system).
    NeoN::Vec3 dragDir_ {1, 0, 0};
    NeoN::Vec3 liftDir_ {0, 0, 1};
    NeoN::Vec3 sideDir_ {0, 1, 0}; // derived in read()

    // Last computed scalar coefficients (set by execute(), consumed by write())
    double Cd_ {0};
    double Cl_ {0};
    double Cs_ {0};
    double CmRoll_ {0};
    double CmPitch_ {0};
    double CmYaw_ {0};
};

} // namespace NeoFOAM
