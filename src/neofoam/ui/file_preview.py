# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What Save would write to the per-field numerics files, through the real writer."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from neofoam.io import write_configs
from neofoam.ui import case_spec as cs
from neofoam.ui.form_schema import NF_FIELD_FORM
from neofoam.ui.forms import FormEntry


def preview_entries(entries: list[FormEntry]) -> list[FormEntry]:
    """Every form writing to a file that has a per-field form, co-owners included."""
    files = {e.cls.io_config.file for e in entries if NF_FIELD_FORM in e.form_schema}
    return [
        entry
        for entry in entries
        if getattr(entry.cls, "io_config", None) is not None and entry.cls.io_config.file in files
    ]


def _invalid_text(error: ValidationError) -> str:
    """A validation error as comment lines in place of the file."""
    problems = [
        f"// {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    ]
    return "\n".join(["// Save would refuse this form:", *problems]) + "\n"


def _written_text(entries: list[FormEntry], spec: dict[str, Any], target: Path | None) -> str:
    """One file as the writer leaves it, starting from the copy on disk like a save does."""
    file: str = entries[0].cls.io_config.file
    with tempfile.TemporaryDirectory() as scratch:
        written = Path(scratch) / file
        if target is not None and (target / file).is_file():
            written.parent.mkdir(parents=True)
            shutil.copyfile(target / file, written)
        configs = [
            entry.cls.model_validate(spec[entry.config_name])
            for entry in entries
            if entry.config_name in spec
        ]
        write_configs(configs, scratch)
        return written.read_text()


def preview_files(
    entries: list[FormEntry],
    form_state: dict[str, dict[str, Any]],
    selected: set[str],
    target: Path | None,
    refused: Collection[str] = (),
) -> dict[str, str]:
    """The text Save would leave per file; a form Save would refuse reads as its errors.

    Call it with :func:`preview_entries`. A ``refused`` file, which Save leaves alone, reads
    as it is on disk. Writes only to a temporary directory, so it is
    safe on every edit; the OpenFOAM dictionary layer is not thread-safe.

    Example::

        texts = preview_files(preview_entries(entries), form_state, selected, target)
    """
    spec = cs.state_to_case_spec(entries, form_state, selected)
    by_file: dict[str, list[FormEntry]] = {}
    for entry in entries:
        if entry.config_name in spec:
            by_file.setdefault(entry.cls.io_config.file, []).append(entry)
    texts: dict[str, str] = {}
    for file, owners in by_file.items():
        if target is not None and file in refused:
            texts[file] = (target / file).read_text()
            continue
        try:
            texts[file] = _written_text(owners, spec, target)
        except ValidationError as error:
            texts[file] = _invalid_text(error)
    return texts
