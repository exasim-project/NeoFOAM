# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""INT-2 / C2 — the SolverSpec builds and enumerates its configs."""

from neofoam.solver.incompressibleFluidBlockAMR import (
    config_classes,
    incompressibleFluidBlockAMR,
)


def test_solver_spec_name():
    assert incompressibleFluidBlockAMR.name == "incompressibleFluidBlockAMR"


def test_config_classes_non_empty_and_enumerates_core_configs():
    classes = config_classes()
    assert len(classes) > 0
    names = {c.__name__ for c in classes}
    assert "MeshDictConfig" in names
    assert "ControlDictConfig" in names
    assert "BlockAMRSolutionConfig" in names
