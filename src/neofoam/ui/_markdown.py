# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The chat's markdown subset (bold, italics, inline code, lists, line breaks) as safe HTML."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from itertools import groupby

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
# Only at word boundaries: a path such as my_case_dir keeps its underscores.
_ITALIC = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_ITEM = re.compile(r"^\s*[-*]\s+(.*)$")


def _inline(line: str) -> str:
    """Bold, italics and inline code of one escaped line; code spans stay literal."""
    parts = _CODE.split(line)
    for i, part in enumerate(parts):
        if i % 2:
            parts[i] = f"<code>{part}</code>"
        else:
            parts[i] = _ITALIC.sub(r"<em>\1</em>", _BOLD.sub(r"<strong>\1</strong>", part))
    return "".join(parts)


def _list(items: Iterable[str]) -> str:
    rows = "".join("<li>" + _inline(_ITEM.sub(r"\1", item)) + "</li>" for item in items)
    # Vuetify's reset takes a list's padding, which holds the bullets.
    return f'<ul style="padding-left: 1.2em;">{rows}</ul>'


def _chat_html(text: str) -> str:
    """``text`` as HTML for ``v-html``: escaped first, so only the tags made here exist."""
    lines = html.escape(text).split("\n")
    blocks = [
        _list(group) if is_list else "<br>".join(_inline(line) for line in group)
        for is_list, group in groupby(lines, key=lambda line: bool(_ITEM.match(line)))
    ]
    return "".join(blocks)
