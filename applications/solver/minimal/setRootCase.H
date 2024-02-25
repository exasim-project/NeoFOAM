// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
// Construct from (int argc, char* argv[]),
// - use argList::argsMandatory() to decide on checking command arguments.
// - check validity of the options

NeoFOAM::argList args(argc, argv);
if (!args.checkRootCase())
{
    NeoFOAM::FatalError.exit();
}

// User can also perform checks on otherwise optional arguments.
// Eg,
//
//  if (!args.check(true, false))
//  {
//      Foam::FatalError.exit();
//  }

// Force dlOpen of FOAM_DLOPEN_LIBS (principally for Windows applications)
// #include "foamDlOpenLibs.H"
