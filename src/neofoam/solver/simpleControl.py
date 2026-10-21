# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""
Pure-Python SIMPLE control, mirroring the logic in OpenFOAM's
Foam::simpleControl / Foam::solutionControl.

Control parameters are passed directly as plain Python values; no C++
bindings are required inside this module.
"""

from __future__ import annotations


class SimpleControl:
    """SIMPLE/SIMPLEC non-orthogonal-corrector controller.

    Parameters are passed directly as plain Python values read from the
    OpenFOAM ``system/fvSolution`` SIMPLE sub-dictionary:

    * ``n_non_orthogonal_correctors`` – non-orthogonal correctors
    * ``momentum_predictor``          – bool flag
    * ``consistent``                  – bool flag (SIMPLEC vs plain SIMPLE)

    Unlike PISO/PIMPLE, SIMPLE has no inner pressure-correction loop (one
    pressure solve per outer iteration), so there is no ``correct()``.
    """

    def __init__(
        self,
        n_non_orthogonal_correctors: int = 0,
        momentum_predictor: bool = True,
        consistent: bool = False,
    ) -> None:
        self._n_non_orth: int = n_non_orthogonal_correctors
        self._momentum_predictor: bool = momentum_predictor
        self._consistent: bool = consistent

        self._corr_non_orth: int = 0

    def momentum_predictor(self) -> bool:
        """Return True when the momentum predictor step should be solved."""
        return self._momentum_predictor

    def consistent(self) -> bool:
        """Return True for SIMPLEC (consistent rAtU) rather than plain SIMPLE."""
        return self._consistent

    def correct_non_orthogonal(self) -> bool:
        """Advance the non-orthogonal corrector index.

        Call in a ``while simple.correct_non_orthogonal():`` loop. Returns
        True for (nNonOrthogonalCorrectors + 1) passes, then False.
        """
        self._corr_non_orth += 1
        if self._corr_non_orth <= self._n_non_orth + 1:
            return True
        self._corr_non_orth = 0
        return False

    def final_non_orthogonal_iter(self) -> bool:
        """Return True on the final non-orthogonal corrector iteration."""
        return self._corr_non_orth == self._n_non_orth + 1
