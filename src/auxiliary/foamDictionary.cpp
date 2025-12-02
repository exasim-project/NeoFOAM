// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/auxiliary/convert.hpp"

#include "vector.H"
#include <functional>

namespace NeoFOAM
{

static std::vector<std::function<bool(NeoN::Dictionary&, const Foam::entry&)>> mapEntries = {
    [](NeoN::Dictionary& neoDict, const Foam::entry& entry)
    {
        if (checkEntryType<Foam::label>(entry))
        {
            Foam::token::tokenType tt = entry.stream()[0].type();
            if (tt == Foam::token::tokenType::LABEL)
            {
                neoDict.insert(entry.keyword(), convert(entry.get<Foam::label>()));
            }
            else
            {
                neoDict.insert(entry.keyword(), entry.get<Foam::scalar>());
            }
            return true;
        }
        return false;
    },
    &insert<Foam::scalar>,
    &insert<Foam::vector>,
    &insert<Foam::word>,
    {[](NeoN::Dictionary& neoDict, const Foam::entry& entry)
     {
         if (entry.isStream())
         {
             neoDict.insert(entry.keyword(), convert(entry.stream()));
             return true;
         }
         return false;
     }}
};


void insertEntry(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    std::string keyword = entry.keyword();
    for (auto& mapEntry : mapEntries)
    {
        if (mapEntry(neoDict, entry))
        {
            return;
        }
    }
    std::string entryString;
    if (entry.isStream())
    {
        entryString = entry.stream().toString();
    }
    throw std::runtime_error(
        "No known conversion for the key: " + keyword
        + " \n"
          "and the following entry: "
        + entryString
    );
}

void readFoamDictionary(const Foam::dictionary& dict, NeoN::Dictionary& neoDict)
{
    for (auto& entry : dict)
    {
        std::string keyword = entry.keyword();
        if (entry.isDict())
        {
            NeoN::Dictionary subDict;
            readFoamDictionary(entry.dict(), subDict);
            neoDict.insert(entry.keyword(), subDict);
        }
        else
        {
            insertEntry(neoDict, entry);
        }
    }
}

NeoN::Dictionary convert(const Foam::dictionary& dict)
{
    NeoN::Dictionary neoDict;
    readFoamDictionary(dict, neoDict);
    return neoDict;
}

} // namespace Foam
