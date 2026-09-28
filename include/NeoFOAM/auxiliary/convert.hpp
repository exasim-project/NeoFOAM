// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoN/NeoN.hpp"

#include "fvc.H"

namespace NeoFOAM
{
// To NeoN
NeoN::Vec3 convert(const Foam::vector& in);

NeoN::scalar convert(const Foam::scalar& in);

std::string convert(const Foam::word& in);

NeoN::TokenList convert(const Foam::ITstream& in);

NeoN::label convert(const Foam::label& in);

NeoN::Dictionary convert(const Foam::dictionary dict);

/* @brief The NeoN dictionary key for an OpenFOAM entry.
 *
 * OpenFOAM only treats a *quoted* keyword as a regular expression and the NeoN
 * dictionary has no pattern flag, so a regex keyword keeps its quotes as part of the
 * key. The regex-aware lookups in NeoFOAM/compatibility/fvSolution.hpp rely on that
 * to tell a pattern (`"(U|k|epsilon)"`) from a literal keyword.
 */
std::string dictKey(const Foam::entry& entry);

template<typename T>
bool checkEntryType(const Foam::entry& entry)
{
    Foam::FatalError.throwExceptions(true);
    Foam::FatalIOError.throwExceptions(true);
    try
    {
        // NOTE since get<T> this can cast int -> float or float -> int
        // we need to check whether the underlying token type actually matches
        if constexpr (std::is_same_v<T, NeoN::scalar>)
        {
            if (entry.stream().tokens().size() == 1)
            {
                bool isLabel = entry.stream().tokens()[0].type() == Foam::token::tokenType::LABEL;
                // we are testing whether the entryType is a scalar but the underlying token is
                // a label
                if (isLabel)
                {
                    return false;
                }
            }
        }
        entry.get<T>();
    }
    catch (const Foam::IOerror& ioErr)
    {
        Foam::FatalError.throwExceptions(false);
        Foam::FatalIOError.throwExceptions(false);
        return false;
    }
    catch (const Foam::error& err)
    {
        Foam::FatalError.throwExceptions(false);
        Foam::FatalIOError.throwExceptions(false);
        return false;
    }
    Foam::FatalError.throwExceptions(false);
    Foam::FatalIOError.throwExceptions(false);
    return true;
}

template<typename T>
bool insert(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    if (checkEntryType<T>(entry))
    {
        neoDict.insert(dictKey(entry), convert(entry.get<T>()));
        return true;
    }
    return false;
}

NeoN::TokenList convert(const Foam::ITstream& stream);

// To Foam
Foam::vector convert(const NeoN::Vec3& in);

}; // namespace NeoFOAM
