// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

// NeoN must be included before OpenFOAM headers: OpenFOAM defines a NotImplemented
// macro that conflicts with gko::NotImplemented in the Ginkgo linear algebra library.
#include "NeoN/NeoN.hpp"

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

    // ---- Formatting constants (mirror OF writeFile) ----
    static constexpr int writePrecision = 6;
    static constexpr int charWidth = writePrecision + 8; // = 14

    const Foam::Time& time_;

    //- Return (and create if needed) postProcessing/<name()>/<startTime>/ path
    std::filesystem::path outputDir() const;

    /**
     * @brief Get or create a persistent output file.
     *
     * On the first call the directory is created, the file is opened, and
     * (if provided) @p writeHeaderFn is called to write the file header.
     * On subsequent calls the existing open stream is returned for appending.
     *
     * @param filename       File name inside outputDir()
     * @param writeHeaderFn  Called once on file creation to write the header block.
     *                       Pass {} to write no header.
     */
    std::ofstream& getOrCreateFile(
        const std::string& filename,
        std::function<void(std::ostream&)> writeHeaderFn = {}
    );

    // ---- Output formatting helpers (mirror OF writeFile helpers) ----

    //- Write current simulation time in scientific notation
    void writeCurrentTime(std::ostream& os) const;

    //- Format a scalar in scientific notation
    static std::string fmtScalar(double v);

    //- Format a Vec3 as "(x y z)" string for header metadata lines
    static std::string fmtVec3(const NeoN::Vec3& v);

    //- Write "# title\n" — blank title gives just "#\n"
    static void writeHeader(std::ostream& os, const std::string& title);

    //- Write "# name            : value\n" — name left-padded to charWidth-2
    static void
    writeHeaderValue(std::ostream& os, const std::string& name, const std::string& value);

    //- Write "# str" left-padded to charWidth-2, NO newline (caller appends columns)
    static void writeCommented(std::ostream& os, const std::string& str);

    //- Write " str" right-padded to charWidth, NO newline
    static void writeTabbed(std::ostream& os, const std::string& str);

    //- Write one Vec3 as three scalars with field width
    static void writeVec3(std::ostream& os, const NeoN::Vec3& v);

private:

    std::string startTime_;
    std::map<std::string, std::ofstream> files_;
};

} // namespace NeoFOAM
