# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2023 NeoFOAM authors
import json

import networkx as nx  # type: ignore[import-untyped]
from pyvis.network import Network  # type: ignore[import-untyped]


def digraph_to_pyvis_html(G: nx.DiGraph, html_path: str = "dag.html") -> None:
    net = Network(directed=True, notebook=False)
    for node, attrs in G.nodes(data=True):
        shape = attrs.get("shape", "ellipse")  # default if missing
        net.add_node(
            node, label=str(node), shape=shape, color=attrs.get("color", "lightblue")
        )
    for source, target in G.edges:
        net.add_edge(source, target)

    # Layout options
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 180,
                "levelSeparation": 150,
            }
        },
        "physics": {"enabled": True},
        "edges": {
            "arrows": {"to": {"enabled": True}},
            "smooth": False,
            "font": {"size": 14, "align": "middle"},  # edge label styling
        },
    }

    net.set_options(json.dumps(options))
    net.write_html(html_path)
