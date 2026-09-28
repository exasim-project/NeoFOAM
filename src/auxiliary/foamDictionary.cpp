// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/auxiliary/convert.hpp"

#include "vector.H"
#include <functional>

namespace NeoFOAM
{


std::string dictKey(const Foam::entry& entry)
{
    const Foam::keyType& keyword = entry.keyword();
    return keyword.isPattern() ? "\"" + std::string(keyword) + "\"" : std::string(keyword);
}

static bool insertStream(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    if (!entry.isStream()) return false;
    std::any insert = convert(entry.stream());
    neoDict.insert(dictKey(entry), insert);
    return true;
}

static bool insertScalar(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    if (!checkEntryType<Foam::scalar>(entry))
    {
        return false;
    }
    std::any insert = convert(entry.get<Foam::scalar>());
    neoDict.insert(dictKey(entry), insert);
    return true;
}

static bool insertWord(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    if (!checkEntryType<Foam::word>(entry))
    {
        return false;
    }
    std::any insert = convert(entry.get<Foam::word>());
    neoDict.insert(dictKey(entry), insert);
    return true;
}

static bool insertLabel(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    if (!checkEntryType<Foam::label>(entry))
    {
        return false;
    }
    std::any insert = convert(entry.get<Foam::label>());
    neoDict.insert(dictKey(entry), insert);
    return true;
}


/**@brief a vector of possible conversions */
static std::vector<std::function<bool(NeoN::Dictionary&, const Foam::entry&)>>
    foamToNeoNEntryConverters = {
        insertScalar,
        insertLabel,
        insertWord,
        &insert<Foam::vector>,
        insertStream,
};

void insertEntry(NeoN::Dictionary& neoDict, const Foam::entry& entry)
{
    std::string keyword = entry.keyword();
    for (auto& mapEntry : foamToNeoNEntryConverters)
    {
        if (mapEntry(neoDict, entry))
        { // a match has been found return
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
            neoDict.insert(dictKey(entry), subDict);
        }
        else
        {
            insertEntry(neoDict, entry);
        }
    }
}

NeoN::Dictionary convert(const Foam::dictionary dict)
{
    NeoN::Dictionary neoDict;
    readFoamDictionary(dict, neoDict);
    return neoDict;
}

} // namespace Foam
