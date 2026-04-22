// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/functionObjects/forces.hpp"
#include "NeoFOAM/datastructures/databaseWrapper.hpp"

#include "addToRunTimeSelectionTable.H"
#include "polyMesh.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

// Manual RTST registration: TypeName macro cannot be used in namespace NeoFOAM
// because it generates `virtual const word& type()` with unqualified `word`.
const Foam::word NeoFOAM::Forces::typeName("neoForces");
int NeoFOAM::Forces::debug(0);

namespace
{
// Register Forces in Foam::functionObject's RTST at library load time
Foam::functionObject::adddictionaryConstructorToTable<NeoFOAM::Forces>
    addNeoFOAMForcesToRunTimeSelectionTable("neoForces");
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace NeoFOAM
{

namespace
{
// Helper: look up the MeshAdapter from the OpenFOAM object registry.
// Throws FatalError if the registered fvMesh is not a NeoFOAM::MeshAdapter.
const MeshAdapter& lookupMeshAdapter(const Foam::Time& runTime)
{
    const Foam::fvMesh& foamMesh =
        runTime.lookupObject<Foam::fvMesh>(Foam::polyMesh::defaultRegion);
    const MeshAdapter* adapter = dynamic_cast<const MeshAdapter*>(&foamMesh);
    if (!adapter)
    {
        Foam::FatalError << "NeoFOAM::Forces requires the registered fvMesh to be a "
                            "NeoFOAM::MeshAdapter, but a plain Foam::fvMesh was found.\n"
                            "Make sure to call NeoFOAM::createAdapterRunTime() before "
                            "instantiating NeoFOAM functionObjects."
                         << Foam::abort(Foam::FatalError);
    }
    return *adapter;
}
} // anonymous namespace


// ---- Constructor ----

Forces::Forces(const Foam::word& name, const Foam::Time& runTime, const Foam::dictionary& dict)
    : FunctionObjectIO(name, runTime, dict)
{
    read(dict);
}


// ---- resolveMesh (lazy) ----

void Forces::resolveMesh()
{
    if (meshAdapter_) return;
    meshAdapter_ = &lookupMeshAdapter(time_);

    // Resolve patch names to indices now that the mesh is available
    patchIndices_.clear();
    const Foam::polyBoundaryMesh& pbm = meshAdapter_->boundaryMesh();
    for (const Foam::word& patchName : patchNames_)
    {
        const Foam::label idx = pbm.findPatchID(patchName);
        if (idx < 0)
        {
            Foam::FatalError << "Patch '" << patchName << "' not found in mesh boundary.\n"
                             << "Available patches: " << pbm.names()
                             << Foam::abort(Foam::FatalError);
        }
        patchIndices_.push_back(static_cast<int>(idx));
    }

    // Read nu once — resolveMesh() is re-entered only when meshAdapter_ is reset (dict change).
    // transportProperties is registered in the mesh's objectRegistry, not in Time.
    nu_ = 0.0;
    if (meshAdapter_->foundObject<Foam::IOdictionary>("transportProperties"))
    {
        const auto& props = meshAdapter_->lookupObject<Foam::IOdictionary>("transportProperties");
        nu_ = props.get<Foam::scalar>("nu");
    }

    namespace fvcc = NeoN::finiteVolume::cellCentred;
    const NeoN::Executor& exec = meshAdapter_->exec();
    const NeoN::UnstructuredMesh& nfMesh = meshAdapter_->nfMesh();

    gradOp_ = std::make_unique<fvcc::GaussGreenGrad>(exec, nfMesh);
    gradU_ = std::make_unique<fvcc::VolumeField<NeoN::Tensor>>(
        exec,
        "gradU",
        nfMesh,
        fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Tensor>>(nfMesh)
    );
}


// ---- read ----

bool Forces::read(const Foam::dictionary& dict)
{
    pName_ = dict.getOrDefault<Foam::word>("pName", "p");
    uName_ = dict.getOrDefault<Foam::word>("uName", "U");
    rhoRef_ = dict.getOrDefault<Foam::scalar>("rhoInf", 1.0);
    pRef_ = dict.getOrDefault<Foam::scalar>("pRef", 0.0);

    Foam::vector cofRFoam = dict.getOrDefault<Foam::vector>("CofR", Foam::vector::zero);
    cofR_ = NeoN::Vec3(cofRFoam[0], cofRFoam[1], cofRFoam[2]);

    // Store patch names; index resolution is deferred to first execute()
    // (the MeshAdapter may not be registered yet at construction time)
    patchNames_ = dict.get<Foam::wordList>("patches");
    meshAdapter_ = nullptr; // force re-resolve if dict changes
    patchIndices_.clear();
    gradOp_.reset();
    gradU_.reset();

    return true;
}


// ---- GPU kernel ----

void Forces::computePatchForces(
    int patchi,
    const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::scalar>& nfP,
    NeoN::scalar rhoRef,
    NeoN::scalar pRef,
    const NeoN::Vec3& cofR,
    ForceResult& result
) const
{
    // Range of boundary-face indices for this patch
    auto [start, end] = nfP.boundaryData().range(patchi);

    const NeoN::UnstructuredMesh& nfMesh = meshAdapter_->nfMesh();
    const NeoN::Executor exec = meshAdapter_->exec();

    // Device views — no host copy of the full field
    auto sfView = nfMesh.boundaryMesh().sf().view();
    auto cfView = nfMesh.boundaryMesh().cf().view();
    auto pBcView = nfP.boundaryData().value().view();

    // 6-element device accumulator initialised to zero
    // Layout: [fp.x, fp.y, fp.z, mp.x, mp.y, mp.z]
    NeoN::Vector<NeoN::scalar> acc(exec, 6, NeoN::scalar {0});
    auto accView = acc.view();

    // GPU-portable kernel: compute pressure force/moment per face
    NeoN::parallelFor(
        exec,
        {static_cast<NeoN::localIdx>(start), static_cast<NeoN::localIdx>(end)},
        NEON_LAMBDA(const NeoN::localIdx bfacei) {
            // Pressure force on this face: F = rho * (p - pRef) * Sf
            NeoN::Vec3 fp = rhoRef * (pBcView[bfacei] - pRef) * sfView[bfacei];

            // Lever arm from centre of rotation to face centre
            NeoN::Vec3 lv = cfView[bfacei] - cofR;

            // Moment: lv × fp (inline cross product — no NeoN::cross yet)
            NeoN::Vec3 mp(
                lv[1] * fp[2] - lv[2] * fp[1],
                lv[2] * fp[0] - lv[0] * fp[2],
                lv[0] * fp[1] - lv[1] * fp[0]
            );

            // Atomic accumulation into device buffer
            NeoN::atomic_add(&accView[0], fp[0]);
            NeoN::atomic_add(&accView[1], fp[1]);
            NeoN::atomic_add(&accView[2], fp[2]);
            NeoN::atomic_add(&accView[3], mp[0]);
            NeoN::atomic_add(&accView[4], mp[1]);
            NeoN::atomic_add(&accView[5], mp[2]);
        },
        "Forces::computePatchForces"
    );

    // Transfer only 6 scalars (96 bytes) to host — O(1) regardless of mesh size
    NeoN::Vector<NeoN::scalar> hostAcc = acc.copyToHost();
    auto hv = hostAcc.view();

    result.pressureForce += NeoN::Vec3 {hv[0], hv[1], hv[2]};
    result.pressureMoment += NeoN::Vec3 {hv[3], hv[4], hv[5]};
}


// ---- GPU kernel: viscous forces ----

void Forces::computePatchViscousForces(
    int patchi,
    const NeoN::finiteVolume::cellCentred::VolumeField<NeoN::Tensor>& gradU,
    NeoN::scalar nuRho,
    const NeoN::Vec3& cofR,
    ForceResult& result
) const
{
    auto [start, end] = gradU.boundaryData().range(patchi);

    const NeoN::UnstructuredMesh& nfMesh = meshAdapter_->nfMesh();
    const NeoN::Executor exec = meshAdapter_->exec();

    auto sfView = nfMesh.boundaryMesh().sf().view();
    auto cfView = nfMesh.boundaryMesh().cf().view();
    auto gradUBView = gradU.boundaryData().value().view();

    NeoN::Vector<NeoN::scalar> acc(exec, 6, NeoN::scalar {0});
    auto accView = acc.view();

    NeoN::parallelFor(
        exec,
        {static_cast<NeoN::localIdx>(start), static_cast<NeoN::localIdx>(end)},
        NEON_LAMBDA(const NeoN::localIdx bfacei) {
            NeoN::SymmTensor tau = NeoN::devTwoSymm(gradUBView[bfacei]) * (-nuRho);

            NeoN::Vec3 fv = tau & sfView[bfacei];

            NeoN::Vec3 lv = cfView[bfacei] - cofR;
            NeoN::Vec3 mv(
                lv[1] * fv[2] - lv[2] * fv[1],
                lv[2] * fv[0] - lv[0] * fv[2],
                lv[0] * fv[1] - lv[1] * fv[0]
            );

            NeoN::atomic_add(&accView[0], fv[0]);
            NeoN::atomic_add(&accView[1], fv[1]);
            NeoN::atomic_add(&accView[2], fv[2]);
            NeoN::atomic_add(&accView[3], mv[0]);
            NeoN::atomic_add(&accView[4], mv[1]);
            NeoN::atomic_add(&accView[5], mv[2]);
        },
        "Forces::computePatchViscousForces"
    );

    NeoN::Vector<NeoN::scalar> hostAcc = acc.copyToHost();
    auto hv = hostAcc.view();
    result.viscousForce += NeoN::Vec3 {hv[0], hv[1], hv[2]};
    result.viscousMoment += NeoN::Vec3 {hv[3], hv[4], hv[5]};
}


// ---- execute ----

bool Forces::execute()
{
    if (Foam::Pstream::parRun())
    {
        Foam::FatalError << "NeoFOAM::Forces does not support MPI parallel execution.\n"
                         << "Run in serial or with a single MPI rank."
                         << Foam::abort(Foam::FatalError);
    }

    resolveMesh();

    result_ = ForceResult {};

    namespace fvcc = NeoN::finiteVolume::cellCentred;
    using ScalarField = fvcc::VolumeField<NeoN::scalar>;

    if (!time_.foundObject<DatabaseWrapper>(DatabaseWrapper::registryName))
    {
        WarningInFunction
            << "NeoFOAM database not registered in Foam::Time — skipping Forces::execute()\n"
            << "Ensure createAdapterRunTime() was called before using neoForces." << Foam::endl;
        return false;
    }

    const NeoN::Database& db =
        time_.lookupObject<DatabaseWrapper>(DatabaseWrapper::registryName).db();

    if (!db.contains("VectorCollection"))
    {
        WarningInFunction << "VectorCollection not found in database — skipping Forces::execute()"
                          << Foam::endl;
        return false;
    }

    const fvcc::VectorCollection& vc = fvcc::VectorCollection::instance(db, "VectorCollection");

    auto ids =
        vc.find([&](const NeoN::Document& doc) { return doc.get<std::string>("name") == pName_; });

    if (ids.empty())
    {
        WarningInFunction << "Pressure field '" << pName_
                          << "' not found in VectorCollection — skipping Forces::execute()"
                          << Foam::endl;
        return false;
    }

    const ScalarField& nfP = vc.fieldDoc(ids[0]).field<ScalarField>();

    for (int patchi : patchIndices_)
    {
        computePatchForces(patchi, nfP, rhoRef_, pRef_, cofR_, result_);
    }

    using VecVolumeField = fvcc::VolumeField<NeoN::Vec3>;
    auto uIds =
        vc.find([&](const NeoN::Document& doc) { return doc.get<std::string>("name") == uName_; });

    if (!uIds.empty())
    {
        const VecVolumeField& nfU = vc.fieldDoc(uIds[0]).field<VecVolumeField>();
        gradOp_->gradTensor(nfU, *gradU_);
        const NeoN::scalar nuRho = nu_ * rhoRef_;
        for (int patchi : patchIndices_)
            computePatchViscousForces(patchi, *gradU_, nuRho, cofR_, result_);
    }

    return true;
}


// ---- write ----

bool Forces::write()
{
    const NeoN::Vec3 totalForce = result_.pressureForce + result_.viscousForce;
    const NeoN::Vec3 totalMoment = result_.pressureMoment + result_.viscousMoment;

    // ---- force.dat ----
    {
        auto& fos = getOrCreateFile(
            "force.dat",
            [&](std::ostream& os)
            {
                writeHeader(os, "Force");
                writeHeaderValue(os, "CofR", fmtVec3(cofR_));
                writeHeader(os, "");
                writeCommented(os, "Time");
                for (const auto* col :
                     {"total_x",
                      "total_y",
                      "total_z",
                      "pressure_x",
                      "pressure_y",
                      "pressure_z",
                      "viscous_x",
                      "viscous_y",
                      "viscous_z"})
                {
                    writeTabbed(os, col);
                }
                os << '\n';
            }
        );

        writeCurrentTime(fos);
        writeVec3(fos, totalForce);
        writeVec3(fos, result_.pressureForce);
        writeVec3(fos, result_.viscousForce);
        fos << '\n';
        fos.flush();
    }

    // ---- moment.dat ----
    {
        auto& mos = getOrCreateFile(
            "moment.dat",
            [&](std::ostream& os)
            {
                writeHeader(os, "Moment");
                writeHeaderValue(os, "CofR", fmtVec3(cofR_));
                writeHeader(os, "");
                writeCommented(os, "Time");
                for (const auto* col :
                     {"total_x",
                      "total_y",
                      "total_z",
                      "pressure_x",
                      "pressure_y",
                      "pressure_z",
                      "viscous_x",
                      "viscous_y",
                      "viscous_z"})
                {
                    writeTabbed(os, col);
                }
                os << '\n';
            }
        );

        writeCurrentTime(mos);
        writeVec3(mos, totalMoment);
        writeVec3(mos, result_.pressureMoment);
        writeVec3(mos, result_.viscousMoment);
        mos << '\n';
        mos.flush();
    }

    return true;
}

} // namespace NeoFOAM
