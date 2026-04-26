"""Summary statistics for a built graph.

Read-only diagnostic surface: load a graph.json (or take a NetworkX graph
directly) and produce a structured dict of counts, distributions, and top
hubs. Used by the `graphify stats` CLI subcommand.

No LLM, no network, no writes.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


_CONFIDENCE_LEVELS = ("EXTRACTED", "INFERRED", "AMBIGUOUS")


def load_graph(graph_path: Path) -> nx.Graph:
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    try:
        return json_graph.node_link_graph(raw, edges="links")
    except TypeError:
        return json_graph.node_link_graph(raw)


def compute_stats(G: nx.Graph, top_n: int = 5) -> dict[str, Any]:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Density: 0 for graphs with <2 nodes (NetworkX returns 0 too, but be explicit).
    density = nx.density(G) if n_nodes > 1 else 0.0

    confidence_counts = Counter(
        d.get("confidence", "EXTRACTED") for _, _, d in G.edges(data=True)
    )
    confidence: dict[str, dict[str, float]] = {}
    for level in _CONFIDENCE_LEVELS:
        c = confidence_counts.get(level, 0)
        pct = (c / n_edges * 100.0) if n_edges else 0.0
        confidence[level] = {"count": c, "pct": round(pct, 1)}

    community_counts = Counter(
        d.get("community") for _, d in G.nodes(data=True) if d.get("community") is not None
    )
    n_communities = len(community_counts)
    largest_community = max(community_counts.values(), default=0)

    file_type_counts = Counter(
        d.get("file_type") or "unknown" for _, d in G.nodes(data=True)
    )

    # Top hubs by degree. Tie-break by label for deterministic output.
    hubs: list[dict[str, Any]] = []
    if n_nodes:
        ranked = sorted(
            G.nodes(data=True),
            key=lambda nd: (-G.degree(nd[0]), str(nd[1].get("label", nd[0]))),
        )[: max(0, top_n)]
        hubs = [
            {
                "id": nid,
                "label": data.get("label", nid),
                "degree": G.degree(nid),
                "file_type": data.get("file_type") or "",
            }
            for nid, data in ranked
        ]

    isolates = sum(1 for n in G.nodes() if G.degree(n) == 0)

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": round(density, 6),
        "isolated_nodes": isolates,
        "communities": n_communities,
        "largest_community_size": largest_community,
        "confidence": confidence,
        "file_types": dict(file_type_counts.most_common()),
        "top_hubs": hubs,
    }


def format_stats(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Graph statistics")
    lines.append("================")
    lines.append(f"Nodes:                 {stats['nodes']}")
    lines.append(f"Edges:                 {stats['edges']}")
    lines.append(f"Density:               {stats['density']}")
    lines.append(f"Isolated nodes:        {stats['isolated_nodes']}")
    lines.append(f"Communities:           {stats['communities']}")
    lines.append(f"Largest community:     {stats['largest_community_size']} nodes")

    lines.append("")
    lines.append("Edge confidence:")
    for level in _CONFIDENCE_LEVELS:
        c = stats["confidence"][level]
        lines.append(f"  {level:<10} {c['count']:>6}  ({c['pct']}%)")

    if stats["file_types"]:
        lines.append("")
        lines.append("File types:")
        for ftype, count in stats["file_types"].items():
            lines.append(f"  {ftype:<14} {count}")

    if stats["top_hubs"]:
        lines.append("")
        lines.append(f"Top {len(stats['top_hubs'])} hubs (by degree):")
        for i, h in enumerate(stats["top_hubs"], 1):
            ftype = f" [{h['file_type']}]" if h["file_type"] else ""
            lines.append(f"  {i}. {h['label']} - {h['degree']} edges{ftype}")

    return "\n".join(lines)
