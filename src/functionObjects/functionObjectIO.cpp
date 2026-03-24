// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/functionObjects/functionObjectIO.hpp"

namespace NeoFOAM
{

FunctionObjectIO::FunctionObjectIO(
    const Foam::word& name,
    const Foam::Time& runTime,
    const Foam::dictionary&
)
    : Foam::functionObject(name),
      time_(runTime),
      startTime_(runTime.timeName())
{}

bool FunctionObjectIO::read(const Foam::dictionary&)
{
    return true;
}

bool FunctionObjectIO::end()
{
    return true;
}

std::filesystem::path FunctionObjectIO::outputDir() const
{
    return std::filesystem::path("postProcessing")
        / std::string(this->name())
        / startTime_;
}

std::ofstream& FunctionObjectIO::getOrCreateFile(
    const std::string& filename,
    const std::string& header
)
{
    auto it = files_.find(filename);
    if (it == files_.end())
    {
        auto dir = outputDir();
        std::filesystem::create_directories(dir);
        auto path = dir / filename;
        auto& os = files_[filename];
        os.open(path, std::ios::app);
        if (!os.is_open())
        {
            Foam::FatalError << "Cannot open output file " << path.string()
                             << Foam::abort(Foam::FatalError);
        }
        // Write header only if the file is new (empty)
        if (std::filesystem::file_size(path) == 0)
        {
            os << header << "\n";
        }
        return os;
    }
    return it->second;
}

} // namespace NeoFOAM
