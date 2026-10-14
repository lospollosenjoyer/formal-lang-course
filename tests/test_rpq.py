from collections import deque
from copy import deepcopy
from pathlib import Path
from random import Random

import pytest
from networkx import MultiDiGraph
from networkx.utils import graphs_equal

from project import graph_utils
from project.rpq import ms_bfs_based_rpq, tensor_based_rpq


@pytest.fixture(params=[tensor_based_rpq, ms_bfs_based_rpq], ids=["tensor", "ms-bfs"])
def rpq(request):
    return request.param


def _rpq_by_traversal(
    transitions: dict[int, dict[str, int]],
    accepting: set[int],
    graph: MultiDiGraph,
    starts: set[int],
    finals: set[int],
) -> set[tuple[int, int]]:
    """The supplied DFA has initial state 0."""
    allowed_starts = starts or set(graph.nodes)
    allowed_finals = finals or set(graph.nodes)
    result = set()

    for source in allowed_starts:
        visited = {(source, 0)}
        queue = deque(visited)
        while queue:
            vertex, state = queue.popleft()
            if vertex in allowed_finals and state in accepting:
                result.add((source, vertex))
            for _, target, label in graph.out_edges(vertex, data="label"):
                next_state = transitions.get(state, {}).get(label)
                if next_state is None:
                    continue
                pair = (target, next_state)
                if pair not in visited:
                    visited.add(pair)
                    queue.append(pair)

    return result


@pytest.fixture
def path_graph():
    graph = MultiDiGraph()
    graph.add_nodes_from([100, -7, 42, 999])
    graph.add_edge(100, -7, label="a")
    graph.add_edge(-7, 42, label="b")
    return graph


@pytest.mark.parametrize(
    ("regex", "starts", "finals", "expected"),
    [
        pytest.param("a b", {100}, {42}, {(100, 42)}, id="original-vertex-identifiers"),
        pytest.param("a b", {-7}, {42}, set(), id="start-restriction"),
        pytest.param("a b", {100}, {-7}, set(), id="final-restriction"),
        pytest.param("b a", {100}, {42}, set(), id="label-order"),
        pytest.param("b a", {42}, {100}, set(), id="edge-direction"),
        pytest.param("a | b", set(), {42}, {(-7, 42)}, id="default-starts"),
        pytest.param("a | b", {100}, set(), {(100, -7)}, id="default-finals"),
        pytest.param(
            "a | b", set(), set(), {(100, -7), (-7, 42)}, id="default-both-endpoints"
        ),
        pytest.param(
            "epsilon",
            {100, 999},
            {42, 999},
            {(999, 999)},
            id="restricted-zero-length-paths",
        ),
        pytest.param(
            "a*",
            {999},
            {999},
            {(999, 999)},
            id="isolated-vertex-in-graph-with-edges",
        ),
        pytest.param(
            "a", {999}, {999}, set(), id="isolated-vertex-rejects-nonempty-word"
        ),
        pytest.param("", set(), set(), set(), id="empty-language"),
    ],
)
def test_rpq_paths(rpq, path_graph, regex, starts, finals, expected):
    assert rpq(regex, path_graph, starts, finals) == expected


@pytest.mark.parametrize(
    ("has_loop", "expected"),
    [
        pytest.param(True, {(100, 42)}, id="traverses-loop"),
        pytest.param(False, set(), id="rejects-shorter-path-without-loop"),
    ],
)
def test_rpq_requires_loop_for_last_symbol(rpq, path_graph, has_loop, expected):
    if has_loop:
        path_graph.add_edge(42, 42, label="b")

    assert rpq("a b b", path_graph, {100}, {42}) == expected


@pytest.mark.parametrize("label", ["a", "b"])
def test_rpq_preserves_parallel_edge_labels(rpq, label):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="b")

    assert rpq(label, graph, {10}, {20}) == {(10, 20)}


def test_rpq_returns_all_reachable_endpoint_pairs(rpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(20, 40, label="a")

    result = rpq("a", graph, {10, 20}, {30, 40})

    assert result == {(10, 30), (20, 30), (20, 40)}


def test_rpq_uses_all_accepting_regex_states(rpq, path_graph):
    result = rpq("a | a b", path_graph, {100}, {-7, 42})

    assert result == {(100, -7), (100, 42)}


def test_rpq_empty_graph(rpq):
    assert rpq("a*", MultiDiGraph(), set(), set()) == set()


@pytest.mark.parametrize(
    ("starts", "finals"),
    [
        pytest.param({1234}, {42}, id="unknown-start"),
        pytest.param({100}, {1234}, id="unknown-final"),
    ],
)
def test_rpq_rejects_unknown_vertices(rpq, path_graph, starts, finals):
    with pytest.raises(ValueError):
        rpq("a*", path_graph, starts, finals)


def test_rpq_multicharacter_labels(rpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="first")
    graph.add_edge(20, 30, label="second")

    assert rpq("first second", graph, {10}, {30}) == {(10, 30)}


def test_rpq_does_not_modify_graph(rpq, path_graph):
    path_graph.graph["description"] = ["input graph"]
    path_graph.nodes[100]["annotations"] = ["source"]
    before = deepcopy(path_graph)

    rpq("a b*", path_graph, {100}, {42})

    assert graphs_equal(path_graph, before)


@pytest.mark.parametrize("endpoint", ["start_nodes", "final_nodes"])
def test_rpq_does_not_modify_endpoint_set(rpq, path_graph, endpoint):
    endpoints = {"start_nodes": {100}, "final_nodes": {42}}
    before = endpoints[endpoint].copy()

    rpq("a b*", path_graph, **endpoints)

    assert endpoints[endpoint] == before


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    ("regex", "transitions", "accepting"),
    [
        pytest.param("a b", {0: {"a": 1}, 1: {"b": 2}}, {2}, id="concatenation"),
        pytest.param("a | b", {0: {"a": 1, "b": 1}}, {1}, id="alternative"),
        pytest.param("a*", {0: {"a": 0}}, {0}, id="repetition"),
        pytest.param("(a b)*", {0: {"a": 1}, 1: {"b": 0}}, {0}, id="repeated-sequence"),
        pytest.param(
            "a (b | c)*",
            {0: {"a": 1}, 1: {"b": 1, "c": 1}},
            {1},
            id="alternative-in-repetition",
        ),
    ],
)
def test_rpq_matches_traversal_on_generated_graphs(
    rpq, seed, regex, transitions, accepting
):
    rng = Random(seed)
    nodes = [-10, 8, 42, 100]
    graph = MultiDiGraph()
    graph.add_nodes_from(nodes)
    for source in nodes:
        for target in nodes:
            for label in ["a", "b", "c"]:
                if rng.random() < 0.25:
                    graph.add_edge(source, target, label=label)
    starts = {node for node in nodes if rng.random() < 0.5}
    finals = {node for node in nodes if rng.random() < 0.5}
    expected = _rpq_by_traversal(transitions, accepting, graph, starts, finals)

    assert rpq(regex, graph, starts, finals) == expected, (
        seed,
        regex,
        starts,
        finals,
        list(graph.edges(data="label")),
    )


def test_rpq_keeps_sources_after_paths_merge_at_different_depths(rpq):
    graph = MultiDiGraph()
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 50, label="a")
    graph.add_edge(50, 30, label="a")
    graph.add_edge(30, 30, label="a")
    graph.add_edge(30, 40, label="b")

    assert rpq("a* b", graph, {10, 20}, {40}) == {(10, 40), (20, 40)}


def test_rpq_combines_paths_entering_same_query_state(rpq):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 2, label="b")
    graph.add_edge(1, 3, label="c")
    graph.add_edge(2, 4, label="d")

    assert rpq("a c | b d", graph, {0}, {3, 4}) == {(0, 3), (0, 4)}


def test_rpq_retains_answers_from_earlier_bfs_layers(rpq):
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="a")

    assert rpq("a*", graph, {0}, {1, 2}) == {(0, 1), (0, 2)}


@pytest.mark.parametrize(
    ("first_size", "regex"),
    [
        pytest.param(2, "a a a b b", id="positive-cycle-size"),
        pytest.param(0, "a b b", id="zero-cycle-size-is-a-loop"),
    ],
)
def test_rpq_on_two_cycles(rpq, first_size, regex):
    graph = graph_utils.create_two_cycles_graph(first_size, "a", 1, "b")

    assert rpq(regex, graph, set(), {0}) == {(0, 0)}


def test_rpq_on_loaded_graph(rpq, monkeypatch):
    path = Path(__file__).parent / "data" / "graph_with_parallel_edges.csv"
    monkeypatch.setattr(graph_utils.cfpq_data, "download", lambda _: str(path))
    graph = graph_utils.load_graph("local-fixture")

    assert rpq("a b", graph, set(), set()) == {(0, 1), (0, 2)}
