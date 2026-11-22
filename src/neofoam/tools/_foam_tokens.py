# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Tiny token helpers for the mesh-dict configs (parse ⇄ emit OpenFOAM literals).

The OpenFOAM strategy (:mod:`neofoam.io.strategies.openfoam_strategy`) reads
every non-dict entry back as a **space-separated token string** — e.g.
``vertices`` comes back as ``"( ( -20.6 0 -0.5 ) ( 0 0 -0.5 ) )"`` — because
``pybFoam`` tokenises the primitive entry. These helpers walk that flat token
stream with a balanced-bracket reader so the mesh-dict configs
(:class:`~neofoam.tools.block_mesh.BlockMeshDictConfig`) can parse the compound
sections (``vertices`` / ``blocks`` / ``boundary``) into structured fields and
emit them back. They are the read side of the same string-literal trick
``FieldValue`` uses on the write side.
"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Split a pybFoam-normalised entry string into tokens.

    pybFoam emits brackets space-separated (``( a b )``), so a plain whitespace
    split is a faithful tokeniser; a defensive pad keeps it correct even if a
    caller passes un-spaced brackets.
    """
    for bracket in "(){}":
        text = text.replace(bracket, f" {bracket} ")
    return text.split()


def read_group(
    tokens: list[str],
    i: int,
    *,
    open_: str = "(",
    close: str = ")",
) -> tuple[list[str], int]:
    """Read one balanced ``open_ … close`` group starting at ``tokens[i]``.

    Returns the *inner* tokens (nested brackets preserved, the outermost pair
    dropped) and the index just past the closing bracket.

    Raises:
        ValueError: if ``tokens[i]`` is not ``open_`` or the group is unbalanced.
    """
    if i >= len(tokens) or tokens[i] != open_:
        raise ValueError(f"expected {open_!r} at position {i}, got {tokens[i : i + 1]}")
    depth = 0
    inner: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok == open_:
            depth += 1
            if depth == 1:
                continue  # drop the outermost open bracket
        elif tok == close:
            depth -= 1
            if depth == 0:
                return inner, i  # drop the outermost close bracket
        inner.append(tok)
    raise ValueError(f"unbalanced {open_}{close} group")


def num(x: float) -> str:
    """Format a coordinate for an OpenFOAM literal (drop a trailing ``.0``)."""
    f = float(x)
    return str(int(f)) if f == int(f) else repr(f)


def point(v: tuple[float, float, float]) -> str:
    """An OpenFOAM point literal ``(x y z)``."""
    return f"({num(v[0])} {num(v[1])} {num(v[2])})"
