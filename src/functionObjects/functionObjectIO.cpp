// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/functionObjects/functionObjectIO.hpp"

namespace NeoFOAM
{

FunctionObjectIO::
    FunctionObjectIO(const Foam::word& name, const Foam::Time& runTime, const Foam::dictionary&)
    : Foam::functionObject(name)
    , time_(runTime)
    , startTime_(runTime.timeName())
{}

bool FunctionObjectIO::read(const Foam::dictionary&) { return true; }

bool FunctionObjectIO::end() { return true; }

std::filesystem::path FunctionObjectIO::outputDir() const
{
    return std::filesystem::path("postProcessing") / std::string(this->name()) / startTime_;
}

std::ofstream& FunctionObjectIO::getOrCreateFile(
    const std::string& filename,
    std::function<void(std::ostream&)> writeHeaderFn
)
{
    auto it = files_.find(filename);
    if (it == files_.end())
    {
        auto dir = outputDir();
        std::filesystem::create_directories(dir);
        auto path = dir / filename;
        bool isNew = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
        auto& os = files_[filename];
        os.open(path, std::ios::app);
        if (!os.is_open())
        {
            Foam::FatalError << "Cannot open output file " << path.string()
                             << Foam::abort(Foam::FatalError);
        }
        if (isNew && writeHeaderFn)
        {
            writeHeaderFn(os);
        }
        return os;
    }
    return it->second;
}

void FunctionObjectIO::writeCurrentTime(std::ostream& os) const
{
    os << std::setw(charWidth) << std::scientific << std::setprecision(writePrecision)
       << time_.value();
}

std::string FunctionObjectIO::fmtScalar(double v)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(writePrecision) << v;
    return oss.str();
}

std::string FunctionObjectIO::fmtVec3(const NeoN::Vec3& v)
{
    std::ostringstream oss;
    oss << '(' << std::scientific << std::setprecision(writePrecision) << v[0] << ' ' << v[1] << ' '
        << v[2] << ')';
    return oss.str();
}

void FunctionObjectIO::writeHeader(std::ostream& os, const std::string& title)
{
    if (title.empty())
    {
        os << "#\n";
    }
    else
    {
        os << "# " << title << "\n";
    }
}

void FunctionObjectIO::writeHeaderValue(
    std::ostream& os,
    const std::string& name,
    const std::string& value
)
{
    // "# name            : value\n"
    // name field is (charWidth - 2) wide to account for leading "# "
    os << "# " << std::left << std::setw(charWidth - 2) << name << ": " << value << "\n";
    os << std::right; // restore default alignment
}

void FunctionObjectIO::writeCommented(std::ostream& os, const std::string& str)
{
    // "# str" left-aligned in (charWidth - 2) characters, no newline
    os << "# " << std::left << std::setw(charWidth - 2) << str;
    os << std::right;
}

void FunctionObjectIO::writeTabbed(std::ostream& os, const std::string& str)
{
    // Right-aligned in charWidth characters, no newline
    os << std::right << std::setw(charWidth) << str;
}

void FunctionObjectIO::writeVec3(std::ostream& os, const NeoN::Vec3& v)
{
    os << std::scientific << std::setprecision(writePrecision) << std::setw(charWidth) << v[0]
       << std::setw(charWidth) << v[1] << std::setw(charWidth) << v[2];
}

} // namespace NeoFOAM
