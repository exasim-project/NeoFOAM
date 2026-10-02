# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Pure-Python PISO control loop, mirroring the logic in OpenFOAM's
Foam::pisoControl / Foam::solutionControl.

Control parameters are passed directly as plain Python values; no C++
bindings are required inside this module.
"""

from __future__ import annotations


class PisoControl:
    """PISO corrector-loop controller.

    Parameters are passed directly as plain Python values read from
    the OpenFOAM ``system/fvSolution`` PISO sub-dictionary:

    * ``n_correctors``              – number of PISO correctors
    * ``n_non_orthogonal_correctors`` – non-orthogonal correctors
    * ``momentum_predictor``        – bool flag
    """

    def __init__(
        self,
        n_correctors: int = 2,
        n_non_orthogonal_correctors: int = 0,
        momentum_predictor: bool = False,
    ) -> None:
        self._n_corr: int = n_correctors
        self._n_non_orth: int = n_non_orthogonal_correctors
        self._momentum_predictor: bool = momentum_predictor

        # running counters
        self._corr_piso: int = 0
        self._corr_non_orth: int = 0

    # ------------------------------------------------------------------
    # Public interface (mirrors solutionControl / pimpleControl)
    # ------------------------------------------------------------------

    def momentum_predictor(self) -> bool:
        """Return True when the momentum predictor step should be solved."""
        return self._momentum_predictor

    def correct(self) -> bool:
        """Advance the PISO corrector index.

        Call in a ``while piso.correct():`` loop.  Returns True for each
        corrector pass and False once all nCorrectors are done (resetting
        the counter for the next time-step).
        """
        self._corr_piso += 1
        if self._corr_piso <= self._n_corr:
            return True
        self._corr_piso = 0
        return False

    def correct_non_orthogonal(self) -> bool:
        """Advance the non-orthogonal corrector index.

        Call in a ``while piso.correct_non_orthogonal():`` loop.
        Returns True for (nNonOrthogonalCorrectors + 1) passes, then False.
        """
        self._corr_non_orth += 1
        if self._corr_non_orth <= self._n_non_orth + 1:
            return True
        self._corr_non_orth = 0
        return False

    def final_non_orthogonal_iter(self) -> bool:
        """Return True on the final non-orthogonal corrector iteration."""
        return self._corr_non_orth == self._n_non_orth + 1
