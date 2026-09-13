from collections import Counter
from pathlib import Path
from unittest.mock import Mock, sentinel

import networkx as nx
import pydot
import pytest

from project import graph_utils


def _read_dot(path: str | Path) -> pydot.Dot:
    """Read a DOT file containing exactly one directed graph."""
    parsed = pydot.graph_from_dot_file(str(path))
    assert parsed is not None
    assert len(parsed) == 1
    dot_graph = parsed[0]
    assert dot_graph.get_type() == "digraph"
    return dot_graph


def _read_dot_graph(path: str | Path) -> nx.MultiDiGraph:
    """Read a DOT graph into NetworkX, preserving edge labels and multiplicity."""
    dot_graph = _read_dot(path)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node.get_name() for node in dot_graph.get_nodes())
    graph.add_edges_from(
        (edge.get_source(), edge.get_destination(), {"label": edge.get_label()})
        for edge in dot_graph.get_edges()
    )
    return graph


def test_load_graph_passes_downloaded_path_to_reader(monkeypatch):
    download = Mock(return_value=sentinel.path)
    read = Mock(return_value=sentinel.graph)
    monkeypatch.setattr(graph_utils.cfpq_data, "download", download)
    monkeypatch.setattr(graph_utils.cfpq_data, "graph_from_csv", read)

    graph = graph_utils.load_graph("example")

    download.assert_called_once_with("example")
    read.assert_called_once_with(sentinel.path)
    assert graph is sentinel.graph


@pytest.mark.parametrize(
    ("nodes", "edges", "expected"),
    [
        pytest.param([], [], graph_utils.GraphInfo(0, 0, set()), id="empty"),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1, "a"), (0, 1, "a"), (1, 1, "b")],
            graph_utils.GraphInfo(4, 3, {"a", "b"}),
            id="isolated-nodes-and-repeated-labels",
        ),
    ],
)
def test_load_graph_info(monkeypatch, nodes, edges, expected):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((u, v, {"label": label}) for u, v, label in edges)
    load = Mock(return_value=graph)
    monkeypatch.setattr(graph_utils, "load_graph", load)

    info = graph_utils.load_graph_info("example")

    load.assert_called_once_with("example")
    assert info == expected


def test_load_graph_info_reads_local_csv(monkeypatch):
    graph_path = Path(__file__).parent / "data" / "graph_with_parallel_edges.csv"
    download = Mock(return_value=str(graph_path))
    monkeypatch.setattr(graph_utils.cfpq_data, "download", download)

    info = graph_utils.load_graph_info("example")

    download.assert_called_once_with("example")
    assert info == graph_utils.GraphInfo(3, 4, {"a", "b"})


def test_load_graph_info_rejects_unknown_name():
    with pytest.raises(FileNotFoundError):
        graph_utils.load_graph_info("__nonexistent_graph__")


def test_create_two_cycles_delegates_positive_sizes(monkeypatch):
    generate = Mock(return_value=sentinel.graph)
    monkeypatch.setattr(graph_utils.cfpq_data, "labeled_two_cycles_graph", generate)

    graph = graph_utils.create_two_cycles_graph(2, "a", 5, "b")

    generate.assert_called_once_with(2, 5, labels=("a", "b"))
    assert graph is sentinel.graph


@pytest.mark.parametrize(
    (
        "first_size",
        "first_label",
        "second_size",
        "second_label",
        "expected_nodes",
        "expected_edges",
    ),
    [
        pytest.param(
            0,
            "a",
            2,
            "b",
            {0, 1, 2},
            [(0, 0, "a"), (0, 1, "b"), (1, 2, "b"), (2, 0, "b")],
            id="first-cycle-is-a-loop",
        ),
        pytest.param(
            2,
            "a",
            0,
            "b",
            {0, 1, 2},
            [(0, 1, "a"), (1, 2, "a"), (2, 0, "a"), (0, 0, "b")],
            id="second-cycle-is-a-loop",
        ),
        pytest.param(
            0,
            "a",
            0,
            "b",
            {0},
            [(0, 0, "a"), (0, 0, "b")],
            id="two-loops",
        ),
        pytest.param(
            0,
            "a",
            0,
            "a",
            {0},
            [(0, 0, "a"), (0, 0, "a")],
            id="two-identical-loops",
        ),
    ],
)
def test_create_two_cycles_handles_zero_sizes(
    first_size, first_label, second_size, second_label, expected_nodes, expected_edges
):
    graph = graph_utils.create_two_cycles_graph(
        first_size, first_label, second_size, second_label
    )

    assert set(graph.nodes) == expected_nodes
    assert Counter(graph.edges(data="label")) == Counter(expected_edges)


@pytest.mark.parametrize(("first_size", "second_size"), [(-1, 2), (2, -1)])
def test_create_two_cycles_rejects_negative_sizes(first_size, second_size):
    with pytest.raises(ValueError, match="non-negative"):
        graph_utils.create_two_cycles_graph(first_size, "a", second_size, "b")


@pytest.mark.parametrize("path_type", [Path, str], ids=["path", "string"])
def test_save_two_cycles_matches_golden_graph(tmp_path, path_type):
    output_path = tmp_path / "graph.dot"
    expected_path = Path(__file__).parent / "data" / "two_cycles_5_8.dot"

    graph_utils.save_two_cycles_graph(5, "a", 8, "b", path_type(output_path))

    actual = _read_dot_graph(output_path)
    expected = _read_dot_graph(expected_path)

    assert set(actual.nodes) == set(expected.nodes)
    assert Counter(actual.edges(data="label")) == Counter(expected.edges(data="label"))


def test_save_two_cycles_overwrites_existing_file(tmp_path):
    output_path = tmp_path / "graph.dot"
    output_path.write_text("stale content\n" * 1000, encoding="utf-8")

    graph_utils.save_two_cycles_graph(0, "a", 0, "b", output_path)

    assert "stale content" not in output_path.read_text(encoding="utf-8")
    _read_dot(output_path)


def test_save_two_cycles_preserves_identical_loops(tmp_path):
    output_path = tmp_path / "graph.dot"

    graph_utils.save_two_cycles_graph(0, "a", 0, "a", output_path)

    graph = _read_dot(output_path)
    assert {node.get_name() for node in graph.get_nodes()} == {"0"}
    assert Counter(
        (edge.get_source(), edge.get_destination(), edge.get_label())
        for edge in graph.get_edges()
    ) == Counter({("0", "0", '"a"'): 2})


@pytest.mark.parametrize(
    ("label", "expected_dot_label"),
    [
        pytest.param("a\\", r'"a\\"', id="trailing-backslash"),
        pytest.param('a:\\"b', r'"a:\\\"b"', id="colon-backslash-and-quote"),
    ],
)
def test_save_two_cycles_escapes_labels(
    tmp_path, monkeypatch, label, expected_dot_label
):
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label=label)
    graph.add_edge(1, 0, label="plain")
    create = Mock(return_value=graph)
    monkeypatch.setattr(graph_utils, "create_two_cycles_graph", create)
    output_path = tmp_path / "graph.dot"

    graph_utils.save_two_cycles_graph(2, label, 5, "plain", output_path)

    create.assert_called_once_with(2, label, 5, "plain")
    dot_graph = _read_dot(output_path)
    assert Counter(edge.get_label() for edge in dot_graph.get_edges()) == Counter(
        [expected_dot_label, '"plain"']
    )
