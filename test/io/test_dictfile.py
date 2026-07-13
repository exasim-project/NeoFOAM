# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Behaviour of :class:`neofoam.io.DictFile` across every format it dispatches on.

``DictFile`` mirrors the OpenFOAM ``dictionary`` interface (``get[T]``, ``set``,
``remove``, ``subDict``, ``found``, ``write``) but is format-agnostic: the format
(and thus reader/writer) is inferred from the file name. Nested entries are
addressed with a **tuple** of keys -- ``("PIMPLE", "nCorrectors")`` -- while a
bare string names a top-level entry. The same scenario is parameterized over an
OpenFOAM dictionary (no suffix), YAML, and JSON fixtures with identical content.

OpenFOAM is textual, so ``get`` coerces to the requested type the same way for
every backend: ``get[float]("endTime")`` returns ``0.5`` whether the on-disk
leaf was ``0.5`` (JSON/YAML) or ``"0.5"`` (OpenFOAM).
"""

import shutil
from pathlib import Path

import pytest

from neofoam.io import OF, BaseConfig, DictFile, IOStrategy

CONFIGS = Path(__file__).parent / "configs"

# One fixture per format, same logical content (see configs/dictfile_sample*).
SAMPLES = ["dictfile_sample", "dictfile_sample.yaml", "dictfile_sample.json"]


@IOStrategy(OF("fvSolution", subdict="PIMPLE"))
class _PimpleCfg(BaseConfig):
    """A config mapped to the PIMPLE sub-dict; DictFile uses only its subdict."""

    nCorrectors: int
    nNonOrthogonalCorrectors: int


@pytest.fixture(params=SAMPLES)
def path(request, tmp_path: Path) -> Path:
    """Stage one format's fixture into tmp_path so edits never touch the source."""
    dst = tmp_path / request.param
    shutil.copy(CONFIGS / request.param, dst)
    return dst


# -- read side: get / subDict / found ----------------------------------------


def test_get_top_level(path: Path) -> None:
    d = DictFile(path)
    assert d.get[str]("application") == "icoFoam"
    assert d.get[float]("endTime") == pytest.approx(0.5)


def test_get_nested_via_tuple(path: Path) -> None:
    d = DictFile(path)
    assert d.get[int](("PIMPLE", "nCorrectors")) == 1


def test_subDict_exposes_the_same_interface(path: Path) -> None:
    pimple = DictFile(path).subDict("PIMPLE")
    assert pimple.get[int]("nCorrectors") == 1
    assert pimple.get[int]("nNonOrthogonalCorrectors") == 0


def test_found(path: Path) -> None:
    d = DictFile(path)
    assert d.found("endTime")
    assert d.found(("PIMPLE", "nCorrectors"))
    assert not d.found("missing")
    assert not d.found(("PIMPLE", "missing"))


# -- write side: set / remove / write ----------------------------------------


def test_set_top_level_roundtrips(path: Path) -> None:
    d = DictFile(path)
    d.set("endTime", 1.0)
    d.write()
    assert DictFile(path).get[float]("endTime") == pytest.approx(1.0)


def test_set_nested_via_tuple(path: Path) -> None:
    d = DictFile(path)
    d.set(("PIMPLE", "nCorrectors"), 3)
    d.write()
    reread = DictFile(path)
    assert reread.get[int](("PIMPLE", "nCorrectors")) == 3
    assert reread.get[int](("PIMPLE", "nNonOrthogonalCorrectors")) == 0  # sibling kept


def test_set_creates_new_subdict(path: Path) -> None:
    d = DictFile(path)
    d.set("SIMPLE", {"nNonOrthogonalCorrectors": 2})
    d.write()
    assert DictFile(path).get[int](("SIMPLE", "nNonOrthogonalCorrectors")) == 2


def test_subDict_edits_propagate_on_write(path: Path) -> None:
    d = DictFile(path)
    d.subDict("PIMPLE").set("nCorrectors", 5)
    d.write()
    assert DictFile(path).get[int](("PIMPLE", "nCorrectors")) == 5


def test_remove_nested(path: Path) -> None:
    d = DictFile(path)
    d.remove(("PIMPLE", "nNonOrthogonalCorrectors"))
    d.write()
    reread = DictFile(path)
    assert not reread.found(("PIMPLE", "nNonOrthogonalCorrectors"))
    assert reread.found(("PIMPLE", "nCorrectors"))  # sibling survives


def test_remove_top_level(path: Path) -> None:
    d = DictFile(path)
    d.remove("deltaT")
    d.write()
    reread = DictFile(path)
    assert not reread.found("deltaT")
    assert reread.found("endTime")


def test_remove_absent_key_is_noop(path: Path) -> None:
    d = DictFile(path)
    d.remove("neverPresent")  # no raise
    d.write()
    assert DictFile(path).found("endTime")


def test_write_to_explicit_path_leaves_source_untouched(
    path: Path, tmp_path: Path
) -> None:
    # keep the suffix so the destination resolves to the same format
    out = tmp_path / f"out_{path.name}"
    d = DictFile(path)
    d.set("endTime", 2.0)
    d.write(out)
    assert DictFile(out).get[float]("endTime") == pytest.approx(2.0)
    assert DictFile(path).get[float]("endTime") == pytest.approx(0.5)  # source intact


# -- pydantic bridge: fill (dict -> model) / set (model -> dict) -------------


def test_fill_model_uses_declared_subdict(path: Path) -> None:
    # key defaults from the model's io_config subdict ("PIMPLE"); its file is ignored
    cfg = DictFile(path).fill(_PimpleCfg)
    assert cfg.nCorrectors == 1
    assert cfg.nNonOrthogonalCorrectors == 0
    # validated + coerced to int even for the textual OpenFOAM backend
    assert isinstance(cfg.nCorrectors, int)


def test_fill_model_missing_field_raises(path: Path) -> None:
    d = DictFile(path)
    d.remove(("PIMPLE", "nNonOrthogonalCorrectors"))
    d.write()
    with pytest.raises(ValueError):  # pydantic.ValidationError is a ValueError
        DictFile(path).fill(_PimpleCfg)


def test_fill_model_validate_false_allows_incomplete(path: Path) -> None:
    d = DictFile(path)
    d.remove(("PIMPLE", "nNonOrthogonalCorrectors"))
    d.write()
    cfg = DictFile(path).fill(_PimpleCfg, validate=False)  # model_construct, no raise
    assert cfg.nCorrectors == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DictFile(tmp_path / "nope.json")
