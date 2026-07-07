// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

NeoFOAM::Executor exec = createExecutor(runTime.controlDict());
std::unique_ptr<Foam::MeshAdapter> meshPtr = Foam::createMesh(exec, runTime);
Foam::MeshAdapter& mesh = *meshPtr;
