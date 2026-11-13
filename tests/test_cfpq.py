from copy import deepcopy
from pathlib import Path
from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from networkx.utils import graphs_equal
from pyformlang.cfg import CFG

from project import graph_utils
from project.cfpq import hellings_based_cfpq


@pytest.fixture
def path_graph():
    graph = MultiDiGraph()
    graph.add_nodes_from([100, -7, 42, 999])
    graph.add_edge(100, -7, label="a")
    graph.add_edge(-7, 42, label="b")
    return graph


@pytest.mark.parametrize(
    ("grammar", "starts", "finals", "expected"),
    [
        pytest.param(
            "S -> a b", {100}, {42}, {(100, 42)}, id="original-vertex-identifiers"
        ),
        pytest.param("S -> a b", {-7}, {42}, set(), id="start-restriction"),
        pytest.param("S -> a b", {100}, {-7}, set(), id="final-restriction"),
        pytest.param("S -> b a", {100}, {42}, set(), id="label-order"),
        pytest.param("S -> b a", {42}, {100}, set(), id="edge-direction"),
        pytest.param("S -> a | b", None, {42}, {(-7, 42)}, id="none-starts"),
        pytest.param("S -> a | b", {100}, None, {(100, -7)}, id="none-finals"),
        pytest.param("S -> a | b", set(), {42}, {(-7, 42)}, id="empty-starts"),
        pytest.param("S -> a | b", {100}, set(), {(100, -7)}, id="empty-finals"),
        pytest.param(
            "S -> $", {100, 999}, {42, 999}, {(999, 999)}, id="restricted-epsilon-paths"
        ),
        pytest.param("S -> a", {999}, {999}, set(), id="isolated-vertex-nonempty-word"),
        pytest.param("", None, None, set(), id="empty-language"),
        pytest.param("S -> A\nA -> S", None, None, set(), id="nonproductive-recursion"),
        pytest.param("S -> missing", None, None, set(), id="unknown-edge-label"),
    ],
)
def test_hellings_path_queries(path_graph, grammar, starts, finals, expected):
    result = hellings_based_cfpq(CFG.from_text(grammar), path_graph, starts, finals)

    assert result == expected, (grammar, starts, finals)


def test_hellings_defaults_to_all_endpoint_pairs(path_graph):
    assert hellings_based_cfpq(CFG.from_text("S -> a | b"), path_graph) == {
        (100, -7),
        (-7, 42),
    }


def test_hellings_empty_graph():
    assert hellings_based_cfpq(CFG.from_text("S -> $ | a"), MultiDiGraph()) == set()


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_hellings_rejects_unknown_vertices(path_graph, endpoint):
    with pytest.raises(ValueError):
        hellings_based_cfpq(CFG.from_text("S -> $"), path_graph, **{endpoint: {1234}})


def test_hellings_supports_directed_simple_graph():
    graph = DiGraph()
    graph.add_edge(10, 20, label="first")
    graph.add_edge(20, 30, label="second")

    assert hellings_based_cfpq(CFG.from_text("S -> first second"), graph) == {(10, 30)}


def test_hellings_preserves_literal_epsilon_edge_label():
    graph = DiGraph()
    graph.add_edge(10, 20, label="ɛ")

    assert hellings_based_cfpq(CFG.from_text("S -> ɛ"), graph) == {(10, 20)}


def test_hellings_keeps_parallel_edge_labels():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="b")
    graph.add_edge(20, 30, label="c")
    graph.add_edge(20, 40, label="d")

    assert hellings_based_cfpq(CFG.from_text("S -> a c | b d"), graph) == {
        (10, 30),
        (10, 40),
    }


def test_hellings_uses_custom_start_symbol(path_graph):
    cfg = CFG.from_text("Root -> a b\nS -> a | $", start_symbol="Root")

    assert hellings_based_cfpq(cfg, path_graph) == {(100, 42)}


@pytest.mark.parametrize(
    ("grammar", "labels", "expected"),
    [
        pytest.param(
            "S -> L B\nL -> A M\nM -> A A\nA -> a\nB -> b",
            "aaab",
            {(10, 50)},
            id="left-child-derived-later",
        ),
        pytest.param(
            "S -> A R\nR -> A B\nA -> a\nB -> b",
            "aab",
            {(10, 40)},
            id="right-child-derived-later",
        ),
    ],
)
def test_hellings_joins_new_paths_on_both_sides(grammar, labels, expected):
    graph = MultiDiGraph()
    for index, label in enumerate(labels, start=1):
        graph.add_edge(10 * index, 10 * (index + 1), label=label)

    assert hellings_based_cfpq(CFG.from_text(grammar), graph) == expected, (
        grammar,
        labels,
    )


@pytest.mark.parametrize(
    ("grammar", "expected"),
    [
        pytest.param("S -> E A\nE -> $\nA -> a", {(100, -7)}, id="nullable-left-child"),
        pytest.param(
            "S -> A E\nE -> $\nA -> a", {(100, -7)}, id="nullable-right-child"
        ),
        pytest.param(
            "S -> A B\nA -> $\nB -> $",
            {(100, 100), (-7, -7), (42, 42), (999, 999)},
            id="nullable-both-children",
        ),
        pytest.param(
            "S -> E a E b E\nE -> A\nA -> E | $",
            {(100, 42)},
            id="nullable-unit-cycle-in-long-body",
        ),
    ],
)
def test_hellings_composes_nullable_paths(path_graph, grammar, expected):
    assert hellings_based_cfpq(CFG.from_text(grammar), path_graph) == expected


def test_hellings_balanced_query_requires_revisiting_vertices():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")
    graph.add_edge(10, 30, label="b")
    graph.add_edge(30, 40, label="b")

    assert hellings_based_cfpq(
        CFG.from_text("S -> a S b | a b"), graph, {10}, {40}
    ) == {(10, 40)}


def test_hellings_ambiguous_recursion_terminates_on_cycle():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")

    assert hellings_based_cfpq(CFG.from_text("S -> S S | a | $"), graph) == {
        (10, 10),
        (10, 20),
        (20, 10),
        (20, 20),
    }


def test_hellings_does_not_modify_graph(path_graph):
    path_graph.nodes[100]["annotations"] = ["source"]
    before = deepcopy(path_graph)

    hellings_based_cfpq(CFG.from_text("S -> a S b | $"), path_graph)

    assert graphs_equal(path_graph, before)


def test_hellings_does_not_modify_grammar(path_graph):
    cfg = CFG.from_text("S -> A | a S b\nA -> $")
    before = deepcopy(cfg)

    hellings_based_cfpq(cfg, path_graph)

    assert cfg.start_symbol == before.start_symbol
    assert cfg.variables == before.variables
    assert cfg.terminals == before.terminals
    assert cfg.productions == before.productions


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_hellings_does_not_modify_endpoint_set(path_graph, endpoint):
    endpoints = {"start_nodes": {100}, "final_nodes": {42}}
    before = endpoints[endpoint].copy()

    hellings_based_cfpq(CFG.from_text("S -> a b"), path_graph, **endpoints)

    assert endpoints[endpoint] == before


def _balanced_paths_in_dag(graph, starts, finals):
    result = set()
    for source in starts or set(graph.nodes):
        stack = [(source, ())]
        while stack:
            vertex, word = stack.pop()
            half = len(word) // 2
            if word == ("a",) * half + ("b",) * half and (
                not finals or vertex in finals
            ):
                result.add((source, vertex))
            for _, target, label in graph.out_edges(vertex, data="label"):
                stack.append((target, word + (label,)))
    return result


@pytest.mark.parametrize("seed", range(12))
def test_hellings_matches_exhaustive_balanced_paths_on_generated_dags(seed):
    rng = Random(seed)
    nodes = [-10, 8, 42, 100, -3, 777, 9]
    graph = MultiDiGraph()
    graph.add_nodes_from(nodes)
    for index, source in enumerate(nodes):
        for target in nodes[index + 1 :]:
            for label in ["a", "b", "c"]:
                if rng.random() < 0.3:
                    graph.add_edge(source, target, label=label)
    starts = {node for node in nodes if rng.random() < 0.5}
    finals = {node for node in nodes if rng.random() < 0.5}
    expected = _balanced_paths_in_dag(graph, starts, finals)

    actual = hellings_based_cfpq(CFG.from_text("S -> a S b | $"), graph, starts, finals)

    assert actual == expected, (seed, starts, finals, list(graph.edges(data="label")))


def test_hellings_on_two_cycles():
    graph = graph_utils.create_two_cycles_graph(2, "a", 2, "b")

    assert hellings_based_cfpq(CFG.from_text("S -> a S b | a b"), graph, {0}, {0}) == {
        (0, 0)
    }


def test_hellings_on_loaded_graph(monkeypatch):
    path = Path(__file__).parent / "data" / "graph_with_parallel_edges.csv"
    monkeypatch.setattr(graph_utils.cfpq_data, "download", lambda _: str(path))
    graph = graph_utils.load_graph("local-fixture")

    assert hellings_based_cfpq(CFG.from_text("S -> a b"), graph) == {(0, 1), (0, 2)}
