"""End-to-end CLI smoke tests against a synthetic hello-world app.

Builds a real graph.json from a tiny on-disk app, then exercises every
read-only CLI subcommand via subprocess to catch regressions in argument
parsing, dispatch, error handling, and exit codes.

These complement the per-module unit tests by hitting the real entry point
the user actually invokes (`python -m graphify ...`).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from graphify.detect import detect
from graphify.extract import collect_files, extract
from graphify.build import build_from_json
from graphify.cluster import cluster
from graphify.export import to_json


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- helpers ----------

def _write_helloapp(root: Path) -> None:
    """Write a tiny multi-file Python app the extractor can chew on."""
    (root / "main.py").write_text(
        "from utils import format_name\n"
        "\n"
        "class App:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "\n"
        "    def greet(self) -> str:\n"
        "        return f\"hello, {format_name(self.name)}\"\n"
        "\n"
        "    def run(self) -> None:\n"
        "        print(self.greet())\n"
        "\n"
        "def main() -> None:\n"
        "    App(\"world\").run()\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n",
        encoding="utf-8",
    )
    (root / "utils.py").write_text(
        "def format_name(name: str) -> str:\n"
        "    return name.strip().title()\n"
        "\n"
        "def shout(msg: str) -> str:\n"
        "    return msg.upper() + \"!\"\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Hello App\n\nA tiny demo app for graphify smoke tests.\n",
        encoding="utf-8",
    )


def _build_graph(root: Path) -> Path:
    """Run the AST pipeline against `root` and write graph.json. Returns its path."""
    detect(root)
    code_files = collect_files(root)
    extraction = extract(code_files)
    G = build_from_json(extraction)
    communities = cluster(G)
    out_dir = root / "graphify-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.json"
    to_json(G, communities, str(graph_path))
    return graph_path


def _run(args: list[str], cwd: Path, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Invoke `python -m graphify <args>` from `cwd` with a clean HOME."""
    env = os.environ.copy()
    # Point HOME at the temp dir so the version-check in main() is a no-op
    # (no installed skill files there) and per-user state can't leak in.
    env["HOME"] = str(cwd)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "graphify", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture
def app_dir(tmp_path: Path) -> Path:
    _write_helloapp(tmp_path)
    _build_graph(tmp_path)
    return tmp_path


# ---------- pipeline sanity ----------

def test_pipeline_produces_well_formed_graph(app_dir: Path):
    graph_path = app_dir / "graphify-out" / "graph.json"
    assert graph_path.exists()
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert ("links" in data) or ("edges" in data)
    assert len(data["nodes"]) > 0
    labels = {n.get("label") for n in data["nodes"]}
    # Hello-world classes/functions should land in the graph
    assert any(lbl in labels for lbl in ("App", "main", "format_name"))


# ---------- help & dispatch ----------

def test_help_runs_clean(app_dir: Path):
    r = _run(["--help"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "Usage: graphify" in r.stdout
    assert "Commands:" in r.stdout


def test_no_args_prints_usage(app_dir: Path):
    r = _run([], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "Usage: graphify" in r.stdout


def test_unknown_command_fails(app_dir: Path):
    r = _run(["definitely-not-a-command"], cwd=app_dir)
    assert r.returncode != 0
    assert "unknown command" in r.stderr


def test_help_lists_every_dispatched_subcommand(app_dir: Path):
    """Every `elif cmd == "X":` in __main__.py must show up in --help.

    Catches the regression where someone wires a new subcommand but forgets
    to document it in the help block.
    """
    main_py = (REPO_ROOT / "graphify" / "__main__.py").read_text(encoding="utf-8")
    dispatched = set(re.findall(r'elif cmd == "([a-z][a-z0-9-]*)"', main_py))
    # 'install' is the leading `if`, not an `elif` -- add it explicitly.
    dispatched.add("install")
    r = _run(["--help"], cwd=app_dir)
    assert r.returncode == 0
    help_text = r.stdout
    missing = [c for c in dispatched if not re.search(rf"^\s+{re.escape(c)}\b", help_text, re.M)]
    assert not missing, f"subcommands missing from help: {missing}"


# ---------- stats ----------

def test_stats_text_output(app_dir: Path):
    r = _run(["stats"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "Nodes:" in r.stdout
    assert "Edges:" in r.stdout
    assert "Edge confidence:" in r.stdout
    assert "Top" in r.stdout and "hubs" in r.stdout


def test_stats_json_output(app_dir: Path):
    r = _run(["stats", "--json"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    for key in ("nodes", "edges", "density", "communities", "confidence", "top_hubs", "file_types"):
        assert key in data, f"missing key {key}"
    assert data["nodes"] > 0


def test_stats_top_n_limits_hubs(app_dir: Path):
    r = _run(["stats", "--json", "--top", "1"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert len(data["top_hubs"]) <= 1


def test_stats_explicit_graph_path(app_dir: Path):
    r = _run(["stats", "graphify-out/graph.json"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "Nodes:" in r.stdout


def test_stats_missing_file_fails_cleanly(app_dir: Path):
    r = _run(["stats", "no-such-graph.json"], cwd=app_dir)
    assert r.returncode != 0
    assert "graph file not found" in r.stderr


def test_stats_bad_top_flag_fails_cleanly(app_dir: Path):
    r = _run(["stats", "--top", "abc"], cwd=app_dir)
    assert r.returncode != 0
    assert "expects an integer" in r.stderr


def test_stats_unknown_flag_fails_cleanly(app_dir: Path):
    r = _run(["stats", "--bogus"], cwd=app_dir)
    assert r.returncode != 0
    assert "unknown option" in r.stderr


# ---------- explain ----------

def test_explain_known_node(app_dir: Path):
    r = _run(["explain", "App"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "Node:" in r.stdout


def test_explain_unknown_node_does_not_crash(app_dir: Path):
    r = _run(["explain", "ZzzzNopeNope"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    assert "No node matching" in r.stdout


def test_explain_missing_graph_fails(app_dir: Path):
    r = _run(["explain", "App", "--graph", "no-such.json"], cwd=app_dir)
    assert r.returncode != 0
    assert "graph file not found" in r.stderr


# ---------- path ----------

def test_path_between_real_nodes(app_dir: Path):
    r = _run(["path", "App", "format_name"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    # Either a path is shown, or we get the explicit "no path" message.
    assert ("Shortest path" in r.stdout) or ("No path found" in r.stdout)


def test_path_unknown_source_fails_cleanly(app_dir: Path):
    r = _run(["path", "ZzzNoNode", "App"], cwd=app_dir)
    assert r.returncode != 0
    assert "No node matching" in r.stderr


# ---------- query ----------

def test_query_runs_against_graph(app_dir: Path):
    r = _run(["query", "what does App do", "--budget", "500"], cwd=app_dir)
    assert r.returncode == 0, r.stderr
    # query should produce *some* stdout (subgraph or 'no match')
    assert r.stdout.strip() != ""


# ---------- benchmark ----------

def test_benchmark_runs_clean(app_dir: Path):
    r = _run(["benchmark"], cwd=app_dir)
    assert r.returncode == 0, r.stderr


# ---------- check-update ----------

def test_check_update_runs_clean(app_dir: Path):
    r = _run(["check-update", "."], cwd=app_dir)
    assert r.returncode == 0, r.stderr
