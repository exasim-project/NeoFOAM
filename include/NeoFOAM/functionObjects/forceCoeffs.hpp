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
 * Extends Forces by dividing the raw pressure forces and moments by the
 * dynamic pressure reference quantities to produce dimensionless coefficients.
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
 *     }
 * }
 * @endcode
 *
 * @par Output columns (forceCoeffs.dat)
 * @code
 * # Time  Cd.x  Cd.y  Cd.z  Cm.x  Cm.y  Cm.z
 * @endcode
 * where @c Cd = pressureForce / (0.5 * rhoInf * magUInf² * Aref)
 * and   @c Cm = pressureMoment / (0.5 * rhoInf * magUInf² * Aref * lRef).
 *
 * @note Viscous force contributions are zero in v1.
 * @note MPI parallel execution is not yet supported.
 */
class ForceCoeffs : public Forces
{
public:

    // OpenFOAM RTST members (manual definition — see forces.hpp comment).
    static const Foam::word typeName;
    static int debug;
    virtual const Foam::word& type() const override { return typeName; }

    //- Construct from name, Time and dictionary (RTST constructor signature)
    ForceCoeffs(
        const Foam::word& name,
        const Foam::Time& runTime,
        const Foam::dictionary& dict
    );

    //- Destructor
    virtual ~ForceCoeffs() = default;

    virtual bool read(const Foam::dictionary& dict) override;

    /**
     * @brief Run Forces::execute() then normalise the result.
     *
     * The ForceResult from the base class is stored unchanged; normalised
     * coefficients are kept in coeffResult_ for write().
     */
    virtual bool execute() override;

    /**
     * @brief Write normalised coefficients to forceCoeffs.dat.
     */
    virtual bool write() override;

private:

    NeoN::scalar magUInf_ {1.0};
    NeoN::scalar lRef_ {1.0};
    NeoN::scalar Aref_ {1.0};

    /// Last normalised force coefficients (dimensionless)
    ForceResult coeffResult_;

    //- Normalise a force vector: F / (0.5 * rhoInf * Uinf² * Aref)
    NeoN::Vec3 normalizeForce(const NeoN::Vec3& f) const;

    //- Normalise a moment vector: M / (0.5 * rhoInf * Uinf² * Aref * lRef)
    NeoN::Vec3 normalizeMoment(const NeoN::Vec3& m) const;
};

} // namespace NeoFOAM
