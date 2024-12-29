// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/core/database/fieldCollection.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;


struct CreateField
{
    std::string name;
    const NeoFOAM::UnstructuredMesh& mesh;
    NeoFOAM::scalar value = 0;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoFOAM::Document operator()(NeoFOAM::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        fvcc::VolumeField<NeoFOAM::scalar> vf(
            mesh.exec(), name, mesh, bcs, fvcc::DataBaseInfo(db, "", "")
        );
        NeoFOAM::fill(vf.internalField(), value);
        return NeoFOAM::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateFieldDoc
        );
    }
};
