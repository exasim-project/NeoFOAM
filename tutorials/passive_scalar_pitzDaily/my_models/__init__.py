# SPDX-License-Identifier: GPL-3.0-or-later
# Reference implementation that doc/tutorials/02-write-a-plugin-model
# walks the reader through. Imported by Allrun via PYTHONPATH so the
# bundled case is self-contained for doc-build figure rendering.

from . import passive_scalar  # noqa: F401  -- triggers plugin registration
