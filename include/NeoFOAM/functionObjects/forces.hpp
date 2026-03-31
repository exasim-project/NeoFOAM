// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <string>
#include <vector>

#include "NeoN/NeoN.hpp"

#include "fvMesh.H"
#include "volFields.H"

#include "NeoFOAM/functionObjects/functionObjectIO.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"

namespace NeoFOAM
{

/**
 * @struct ForceResult
 * @brief Accumulated force and moment vectors from a Forces execution.
 *
 * In v1 only pressure forces/moments are populated; viscous slots are reserved
 * for future implementation.
 */
struct ForceResult
{
    NeoN::Vec3 pressureForce {0, 0, 0};
    NeoN::Vec3 viscousForce {0, 0, 0};   ///< v1: always zero
    NeoN::Vec3 pressureMoment {0, 0, 0};
    NeoN::Vec3 viscousMoment {0, 0, 0};  ///< v1: always zero
};

/**
 * @class Forces
 * @brief GPU-accelerated boundary-patch force and moment calculator.
 *
 * Registered in OpenFOAM's runtime selection table under the type name
 * @c neoForces.  Can be driven directly from @c controlDict:
 * @code
 * functions
 * {
 *     wallForces
 *     {
 *         type      neoForces;
 *         libs      (NeoFOAM);
 *         patches   (fixedWalls);
 *         pName     p;
 *         rhoInf    1.0;
 *         pRef      0.0;
 *         CofR      (0 0 0);
 *     }
 * }
 * @endcode
 *
 * @par GPU I/O strategy
 * execute() runs a @c parallelFor kernel over the boundary patch faces using
 * @c Kokkos::atomic_add into a 6-element device buffer.  Only those 6 scalars
 * (96 bytes) are copied to host — the full pressure field is never transferred.
 * write() appends one row to @c postProcessing/<name>/0/force.dat.
 *
 * @note Viscous forces are not yet implemented (v1 limitation).
 * @note MPI parallel execution is not yet supported (v1 limitation).
 */
class Forces : public FunctionObjectIO
{
public:

    // OpenFOAM RTST members.
    // TypeName("neoForces") cannot be used in namespace NeoFOAM because the
    // macro generates `virtual const word& type()` with unqualified `word`.
    // We define the same interface manually with explicit Foam:: qualification.
    static const Foam::word typeName;
    static int debug;
    virtual const Foam::word& type() const override { return typeName; }

    //- Construct from name, Time and dictionary (RTST constructor signature)
    Forces(
        const Foam::word& name,
        const Foam::Time& runTime,
        const Foam::dictionary& dict
    );

    //- Destructor
    virtual ~Forces() = default;

    // ---- Foam::functionObject interface ----

    virtual bool read(const Foam::dictionary& dict) override;

    /**
     * @brief Run the GPU force-integral computation.
     *
     * Looks up the @c NeoN::Database via the @c DatabaseWrapper registered in the
     * @c Foam::Time objectRegistry, then finds the pressure field @c pName_ in the
     * @c VectorCollection.  The field must have been registered by the solver (e.g. via
     * @c NeoFOAM::constructAndRegister) before the first @c execute() call.
     * Accumulates pressure forces and moments for each registered patch via a
     * GPU-parallel kernel; results are stored in lastResult().
     */
    virtual bool execute() override;

    /**
     * @brief Write the last computed result to postProcessing/.
     *
     * Appends one row to @c force.dat with columns:
     * @c time Fp.x Fp.y Fp.z Mp.x Mp.y Mp.z
     */
    virtual bool write() override;

    //- Access the result from the most recent execute() call
    const ForceResult& lastResult() const { return result_; }

    /**
     * @brief GPU kernel: accumulate pressure force/moment for one boundary patch.
     *
     * Runs @c parallelFor over faces in the patch range and uses
     * @c Kokkos::atomic_add into a 6-element device accumulator.
     * Transfers 6 scalars to host on completion.
     *
     * @param patchi   Patch index (positionally consistent with NeoN boundary offset)
     * @param nfP      NeoN pressure field (on active executor)
     * @param rhoRef   Reference density [kg/m³]
     * @param pRef     Reference pressure [Pa]
     * @param cofR     Centre of rotation [m]
     * @param result   Accumulates into this result (host-side, per-patch contribution added)
     */
    void computePatchForces(
        int patchi,
        const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& nfP,
        NeoN::scalar rhoRef,
        NeoN::scalar pRef,
        const NeoN::Vec3& cofR,
        ForceResult& result
    ) const;

protected:

    /// Resolve MeshAdapter from the object registry on first execute().
    void resolveMesh();

    /// Lazily resolved on first execute() — null until MeshAdapter is registered.
    const MeshAdapter* meshAdapter_ {nullptr};

    Foam::wordList patchNames_;  ///< Stored from dict; resolved to indices lazily
    std::vector<int> patchIndices_;       ///< Resolved from patchNames_ on first execute()

    std::string pName_ {"p"};
    NeoN::scalar rhoRef_ {1.0};
    NeoN::scalar pRef_ {0.0};
    NeoN::Vec3 cofR_ {0, 0, 0};

    ForceResult result_;
};

} // namespace NeoFOAM
