// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include "functionObject.H"
#include "Time.H"

namespace NeoFOAM
{

/**
 * @class FunctionObjectIO
 * @brief Base class for NeoFOAM functionObjects that produce scalar time-series output.
 *
 * Inherits Foam::functionObject so that OpenFOAM's runtime selection table (RTST)
 * can instantiate derived classes directly from controlDict:
 * @code
 * functions
 * {
 *     myForces
 *     {
 *         type   neoForces;
 *         libs   (NeoFOAM);
 *         ...
 *     }
 * }
 * @endcode
 *
 * Derived classes implement:
 *   - execute() — runs GPU computation each time-step
 *   - write()   — writes the scalar result to postProcessing/ at write intervals
 *
 * @par GPU-aware I/O strategy
 * The intent is that GPU computation in execute() reduces an entire field to O(1) scalars,
 * which are then transferred to host and written to disk by write().  Full field copies
 * to host are never performed inside execute().
 */
class FunctionObjectIO : public Foam::functionObject
{
public:

    //- Construct from name, Time and dictionary.
    //  Matches the signature expected by Foam::functionObject::New (RTST).
    FunctionObjectIO(
        const Foam::word& name,
        const Foam::Time& runTime,
        const Foam::dictionary& dict
    );

    //- Destructor
    virtual ~FunctionObjectIO() = default;

    // ---- Foam::functionObject interface ----

    virtual bool read(const Foam::dictionary& dict) override;

    //- GPU computation — implemented by derived classes
    virtual bool execute() override = 0;

    //- Write scalar results — implemented by derived classes
    virtual bool write() override = 0;

    virtual bool end() override;

protected:

    const Foam::Time& time_;

    //- Return (and create if needed) postProcessing/<name()>/<startTime>/ path
    std::filesystem::path outputDir() const;

    /**
     * @brief Get or create a persistent output file.
     *
     * On the first call the directory is created, the file is opened and
     * @p header is written as the first line.  On subsequent calls the
     * existing open stream is returned for appending.
     *
     * @param filename  File name inside outputDir()
     * @param header    Column-header line written once at file creation
     */
    std::ofstream& getOrCreateFile(
        const std::string& filename,
        const std::string& header
    );

private:

    std::string startTime_;
    std::map<std::string, std::ofstream> files_;
};

} // namespace NeoFOAM
