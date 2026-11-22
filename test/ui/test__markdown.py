# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The chat's markdown subset → HTML (pure strings, no trame, no browser).

The output is bound with ``v-html``, and an assistant reply or a file path is
untrusted: every case with markup in its input asserts the escaped form literally.
"""

from __future__ import annotations

import pytest

from neofoam.ui._markdown import _chat_html


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("**Filled:** A, B", "<strong>Filled:</strong> A, B"),
        ("**Loaded** `/tmp/my_case_dir`", "<strong>Loaded</strong> <code>/tmp/my_case_dir</code>"),
        ("Filled: _nothing_", "Filled: <em>nothing</em>"),
        ("read my_case_dir twice", "read my_case_dir twice"),
        ("`**not bold**`", "<code>**not bold**</code>"),
        ("one\n\ntwo", "one<br><br>two"),
        (
            "Steps:\n- first\n* **second**\nDone",
            'Steps:<ul style="padding-left: 1.2em;"><li>first</li>'
            "<li><strong>second</strong></li></ul>Done",
        ),
    ],
)
def test_chat_html_renders_the_markdown_subset(text, expected):
    assert _chat_html(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("**<b onclick=x>**", "<strong>&lt;b onclick=x&gt;</strong>"),
        ("`<script>`", "<code>&lt;script&gt;</code>"),
        ("- <img src=x>", '<ul style="padding-left: 1.2em;"><li>&lt;img src=x&gt;</li></ul>'),
        ('a "quoted" & b', "a &quot;quoted&quot; &amp; b"),
    ],
)
def test_chat_html_escapes_raw_html(text, expected):
    assert _chat_html(text) == expected
