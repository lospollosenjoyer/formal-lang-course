from copy import deepcopy
from pathlib import Path
from random import Random

import pytest
from networkx import DiGraph, MultiDiGraph
from networkx.utils import graphs_equal
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project import graph_utils
from project.cfpq import hellings_based_cfpq, matrix_based_cfpq


@pytest.fixture(
    params=[
        pytest.param(hellings_based_cfpq, id="hellings"),
        pytest.param(matrix_based_cfpq, id="matrix"),
    ]
)
def cfpq(request):
    return request.param


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
def test_cfpq_path_queries(cfpq, path_graph, grammar, starts, finals, expected):
    result = cfpq(CFG.from_text(grammar), path_graph, starts, finals)

    assert result == expected, (grammar, starts, finals)


def test_cfpq_defaults_to_all_endpoint_pairs(cfpq, path_graph):
    assert cfpq(CFG.from_text("S -> a | b"), path_graph) == {
        (100, -7),
        (-7, 42),
    }


def test_cfpq_empty_graph(cfpq):
    assert cfpq(CFG.from_text("S -> $ | a"), MultiDiGraph()) == set()


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_cfpq_rejects_unknown_vertices(cfpq, path_graph, endpoint):
    with pytest.raises(ValueError):
        cfpq(CFG.from_text("S -> $"), path_graph, **{endpoint: {1234}})


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_cfpq_rejects_unknown_vertices_in_empty_graph(cfpq, endpoint):
    with pytest.raises(ValueError):
        cfpq(CFG.from_text("S -> $"), MultiDiGraph(), **{endpoint: {1234}})


def test_cfpq_supports_directed_simple_graph(cfpq):
    graph = DiGraph()
    graph.add_edge(10, 20, label="first")
    graph.add_edge(20, 30, label="second")

    assert cfpq(CFG.from_text("S -> first second"), graph) == {(10, 30)}


@pytest.mark.parametrize("label", ["ɛ", "epsilon", "$"])
def test_cfpq_preserves_literal_epsilon_edge_label(cfpq, label):
    graph = DiGraph()
    graph.add_edge(10, 20, label=label)
    start = Variable("S")
    cfg = CFG(start_symbol=start, productions=[Production(start, [Terminal(label)])])

    assert cfpq(cfg, graph) == {(10, 20)}


def test_cfpq_keeps_parallel_edge_labels(cfpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="b")
    graph.add_edge(20, 30, label="c")
    graph.add_edge(20, 40, label="d")

    assert cfpq(CFG.from_text("S -> a c | b d"), graph) == {
        (10, 30),
        (10, 40),
    }


def test_cfpq_uses_custom_start_symbol(cfpq, path_graph):
    cfg = CFG.from_text("Root -> a b\nS -> a | $", start_symbol="Root")

    assert cfpq(cfg, path_graph) == {(100, 42)}


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        pytest.param(CFG(), set(), id="no-start-symbol"),
        pytest.param(
            CFG(productions=[Production(Variable(None), [Terminal("a")])]),
            set(),
            id="none-valued-variable-is-not-an-implicit-start",
        ),
        pytest.param(
            CFG(
                start_symbol=Variable(None),
                productions=[Production(Variable(None), [Terminal("a")])],
            ),
            {(100, -7)},
            id="explicit-none-valued-start",
        ),
    ],
)
def test_cfpq_distinguishes_missing_and_none_valued_start(
    cfpq, path_graph, cfg, expected
):
    assert cfpq(cfg, path_graph) == expected


def test_cfpq_distinguishes_same_named_terminal_and_variable(cfpq):
    start, variable = Variable("S"), Variable("X")
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(start, [variable, Terminal("a")]),
            Production(start, [Terminal("X"), Terminal("a")]),
            Production(variable, [Terminal("b")]),
        ],
    )
    graph = DiGraph()
    graph.add_edge(10, 20, label="X")
    graph.add_edge(30, 20, label="b")
    graph.add_edge(20, 40, label="a")

    assert cfpq(cfg, graph) == {(10, 40), (30, 40)}


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
def test_cfpq_joins_new_paths_on_both_sides(cfpq, grammar, labels, expected):
    graph = MultiDiGraph()
    for index, label in enumerate(labels, start=1):
        graph.add_edge(10 * index, 10 * (index + 1), label=label)

    assert cfpq(CFG.from_text(grammar), graph) == expected, (
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
def test_cfpq_composes_nullable_paths(cfpq, path_graph, grammar, expected):
    assert cfpq(CFG.from_text(grammar), path_graph) == expected


def test_cfpq_balanced_query_requires_revisiting_vertices(cfpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")
    graph.add_edge(10, 30, label="b")
    graph.add_edge(30, 40, label="b")

    assert cfpq(CFG.from_text("S -> a S b | a b"), graph, {10}, {40}) == {(10, 40)}


def test_cfpq_ambiguous_recursion_terminates_on_cycle(cfpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 10, label="a")

    assert cfpq(CFG.from_text("S -> S S | a | $"), graph) == {
        (10, 10),
        (10, 20),
        (20, 10),
        (20, 20),
    }


def test_cfpq_does_not_modify_graph(cfpq, path_graph):
    path_graph.nodes[100]["annotations"] = ["source"]
    before = deepcopy(path_graph)

    cfpq(CFG.from_text("S -> a S b | $"), path_graph)

    assert graphs_equal(path_graph, before)


def test_cfpq_does_not_modify_grammar(cfpq, path_graph):
    cfg = CFG.from_text("S -> A | a S b\nA -> $")
    before = deepcopy(cfg)

    cfpq(cfg, path_graph)

    assert cfg.start_symbol == before.start_symbol
    assert cfg.variables == before.variables
    assert cfg.terminals == before.terminals
    assert cfg.productions == before.productions


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_cfpq_does_not_modify_endpoint_set(cfpq, path_graph, endpoint):
    endpoints = {"start_nodes": {100}, "final_nodes": {42}}
    before = endpoints[endpoint].copy()

    cfpq(CFG.from_text("S -> a b"), path_graph, **endpoints)

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
def test_cfpq_matches_exhaustive_balanced_paths_on_generated_dags(cfpq, seed):
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

    actual = cfpq(CFG.from_text("S -> a S b | $"), graph, starts, finals)

    assert actual == expected, (seed, starts, finals, list(graph.edges(data="label")))


def test_cfpq_on_two_cycles(cfpq):
    graph = graph_utils.create_two_cycles_graph(2, "a", 2, "b")

    assert cfpq(CFG.from_text("S -> a S b | a b"), graph, {0}, {0}) == {(0, 0)}


def test_cfpq_on_loaded_graph(cfpq, monkeypatch):
    path = Path(__file__).parent / "data" / "graph_with_parallel_edges.csv"
    monkeypatch.setattr(graph_utils.cfpq_data, "download", lambda _: str(path))
    graph = graph_utils.load_graph("local-fixture")

    assert cfpq(CFG.from_text("S -> a b"), graph) == {(0, 1), (0, 2)}


def test_cfpq_reaches_fixed_point_for_recursive_concatenation(cfpq):
    graph = DiGraph()
    for source in range(16):
        graph.add_edge(source, source + 1, label="a")

    result = cfpq(CFG.from_text("S -> S S | a"), graph)

    assert result == {
        (source, target)
        for source in graph.nodes
        for target in graph.nodes
        if source < target
    }


def test_cfpq_uses_boolean_reachability_for_many_alternative_paths(cfpq):
    graph = DiGraph()
    for middle in range(1, 257):
        graph.add_edge(0, middle, label="a")
        graph.add_edge(middle, 257, label="b")

    assert cfpq(CFG.from_text("S -> a b"), graph) == {(0, 257)}
