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
        std::string keyword = entry.keyword();
        neoDict.insert(keyword, convert(entry.get<T>()));
        return true;
    }
    return false;
}

NeoN::TokenList convert(const Foam::ITstream& stream);

// To Foam
Foam::vector convert(const NeoN::Vec3& in);

}; // namespace NeoFOAM
