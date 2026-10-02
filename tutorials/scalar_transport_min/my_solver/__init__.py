# SPDX-License-Identifier: GPL-3.0-or-later
# Reference implementation that doc/tutorials/03-build-a-solver walks
# the reader through. Imported by Allrun via PYTHONPATH so the bundled
# case is self-contained for doc-build figure rendering.

from neofoam.framework.graph import DAGResolver

from .scalar_transport import scalar_transport


def run(argv: list[str] | None = None) -> None:
    solver = scalar_transport.instantiate(argv=argv or [])
    ctx = solver.initialize()
    builder, model_ops = solver.execution_graph()
    resolver = DAGResolver()
    resolved = resolver.resolve(builder, model_ops)
    resolved.operations.run(ctx)
