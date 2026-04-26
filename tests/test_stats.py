"""Tests for graphify/stats.py."""
from __future__ import annotations

import json

import networkx as nx
from networkx.readwrite import json_graph

from graphify.stats import compute_stats, format_stats, load_graph


def _make_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("n1", label="auth", source_file="auth.py", file_type="py", community=0)
    G.add_node("n2", label="api", source_file="api.py", file_type="py", community=0)
    G.add_node("n3", label="main", source_file="main.py", file_type="py", community=1)
    G.add_node("n4", label="errors", source_file="errors.py", file_type="py", community=1)
    G.add_node("n5", label="db", source_file="db.py", file_type="py", community=2)
    G.add_node("n6", label="readme", source_file="README.md", file_type="md", community=2)
    G.add_node("n7", label="orphan", source_file="x.py", file_type="py", community=3)

    G.add_edge("n1", "n2", relation="calls", confidence="INFERRED")
    G.add_edge("n2", "n3", relation="imports", confidence="EXTRACTED")
    G.add_edge("n2", "n4", relation="uses", confidence="EXTRACTED")
    G.add_edge("n2", "n5", relation="reads", confidence="AMBIGUOUS")
    G.add_edge("n3", "n6", relation="documents", confidence="EXTRACTED")
    return G


# --- compute_stats: shape and counts ---

def test_compute_stats_node_and_edge_counts():
    G = _make_graph()
    s = compute_stats(G)
    assert s["nodes"] == 7
    assert s["edges"] == 5


def test_compute_stats_isolated_nodes_counted():
    G = _make_graph()
    s = compute_stats(G)
    # n7 has no edges
    assert s["isolated_nodes"] == 1


def test_compute_stats_density_is_zero_for_single_node():
    G = nx.Graph()
    G.add_node("only")
    s = compute_stats(G)
    assert s["density"] == 0.0


def test_compute_stats_handles_empty_graph():
    G = nx.Graph()
    s = compute_stats(G)
    assert s["nodes"] == 0
    assert s["edges"] == 0
    assert s["density"] == 0.0
    assert s["communities"] == 0
    assert s["largest_community_size"] == 0
    assert s["top_hubs"] == []
    # Confidence keys still present, all zero
    for level in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
        assert s["confidence"][level] == {"count": 0, "pct": 0.0}


# --- compute_stats: confidence distribution ---

def test_confidence_breakdown_percentages_sum_to_100():
    G = _make_graph()
    s = compute_stats(G)
    total_pct = sum(s["confidence"][lvl]["pct"] for lvl in ("EXTRACTED", "INFERRED", "AMBIGUOUS"))
    assert round(total_pct, 1) == 100.0


def test_confidence_breakdown_counts_match_edges():
    G = _make_graph()
    s = compute_stats(G)
    assert s["confidence"]["EXTRACTED"]["count"] == 3
    assert s["confidence"]["INFERRED"]["count"] == 1
    assert s["confidence"]["AMBIGUOUS"]["count"] == 1


def test_confidence_defaults_to_extracted_when_missing():
    G = nx.Graph()
    G.add_node("a")
    G.add_node("b")
    G.add_edge("a", "b")  # no confidence attr
    s = compute_stats(G)
    assert s["confidence"]["EXTRACTED"]["count"] == 1


# --- compute_stats: communities ---

def test_communities_counted_distinctly():
    G = _make_graph()
    s = compute_stats(G)
    assert s["communities"] == 4
    assert s["largest_community_size"] == 2


def test_nodes_without_community_attr_are_ignored():
    G = nx.Graph()
    G.add_node("x", label="x")  # no community
    G.add_node("y", label="y", community=0)
    s = compute_stats(G)
    assert s["communities"] == 1


# --- compute_stats: top hubs ---

def test_top_hubs_sorted_by_degree_desc():
    G = _make_graph()
    s = compute_stats(G, top_n=3)
    hubs = s["top_hubs"]
    assert len(hubs) == 3
    # n2 has degree 4, the highest
    assert hubs[0]["label"] == "api"
    assert hubs[0]["degree"] == 4
    # Subsequent degrees are non-increasing
    degrees = [h["degree"] for h in hubs]
    assert degrees == sorted(degrees, reverse=True)


def test_top_hubs_respects_top_n_zero():
    G = _make_graph()
    s = compute_stats(G, top_n=0)
    assert s["top_hubs"] == []


def test_top_hubs_deterministic_for_ties():
    # Two nodes tied on degree -> deterministic by label
    G = nx.Graph()
    G.add_node("a", label="alpha", community=0)
    G.add_node("b", label="beta", community=0)
    G.add_node("c", label="gamma", community=0)
    G.add_edge("a", "c")
    G.add_edge("b", "c")  # a and b both degree 1, c degree 2
    s = compute_stats(G, top_n=3)
    labels = [h["label"] for h in s["top_hubs"]]
    assert labels == ["gamma", "alpha", "beta"]


# --- compute_stats: file types ---

def test_file_types_breakdown():
    G = _make_graph()
    s = compute_stats(G)
    assert s["file_types"]["py"] == 6
    assert s["file_types"]["md"] == 1


def test_file_type_unknown_when_missing():
    G = nx.Graph()
    G.add_node("a", label="a")
    s = compute_stats(G)
    assert s["file_types"].get("unknown") == 1


# --- format_stats: text output ---

def test_format_stats_contains_key_sections():
    G = _make_graph()
    s = compute_stats(G)
    out = format_stats(s)
    assert "Graph statistics" in out
    assert "Nodes:" in out
    assert "Edges:" in out
    assert "Edge confidence:" in out
    assert "EXTRACTED" in out
    assert "Top" in out and "hubs" in out
    assert "File types:" in out


def test_format_stats_handles_empty_graph_without_error():
    G = nx.Graph()
    s = compute_stats(G)
    out = format_stats(s)
    assert "Nodes:                 0" in out


# --- load_graph round-trip ---

def test_load_graph_roundtrip(tmp_path):
    G = _make_graph()
    data = json_graph.node_link_data(G, edges="links")
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data))
    G2 = load_graph(p)
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()
    s = compute_stats(G2)
    assert s["nodes"] == 7
    assert s["edges"] == 5


# --- JSON shape ---

def test_stats_dict_is_json_serializable():
    G = _make_graph()
    s = compute_stats(G)
    blob = json.dumps(s)  # must not raise
    parsed = json.loads(blob)
    assert parsed["nodes"] == 7
    assert "confidence" in parsed
    assert "top_hubs" in parsed
