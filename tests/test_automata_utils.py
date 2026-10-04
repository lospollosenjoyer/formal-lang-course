import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State

from project.automata_utils import graph_to_nfa, regex_to_dfa


@pytest.mark.parametrize(
    ("regex", "word"),
    [
        pytest.param("a | b", (), id="union-rejects-empty-word"),
        pytest.param("a | b", ("a", "b"), id="union-is-not-concatenation"),
        pytest.param("a b", ("b", "a"), id="concatenation-preserves-order"),
        pytest.param("a b", ("a",), id="incomplete-word"),
        pytest.param("a b", ("a", "b", "b"), id="extra-symbol"),
        pytest.param("a b*", ("b",), id="star-does-not-make-prefix-optional"),
        pytest.param("(a | b)* a", ("a", "b"), id="required-suffix"),
        pytest.param("(a | b)*", ("c",), id="unknown-symbol"),
    ],
)
def test_regex_to_dfa_rejects_words_outside_the_language(regex, word):
    dfa = regex_to_dfa(regex)

    assert not dfa.accepts(word)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        pytest.param((), True, id="empty-word"),
        pytest.param(("a",), False, id="nonempty-word"),
    ],
)
def test_regex_to_dfa_epsilon_language(word, expected):
    dfa = regex_to_dfa("epsilon")

    assert dfa.accepts(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        pytest.param(("label",), True, id="whole-symbol"),
        pytest.param(tuple("label"), False, id="separate-characters"),
    ],
)
def test_regex_to_dfa_multicharacter_symbol(word, expected):
    dfa = regex_to_dfa("label")

    assert dfa.accepts(word) == expected


@pytest.fixture
def path_graph():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="b")
    return graph


@pytest.mark.parametrize(
    ("starts", "finals", "word"),
    [
        pytest.param({20}, {30}, ("a", "b"), id="wrong-start"),
        pytest.param({10}, {20}, ("a", "b"), id="wrong-final"),
        pytest.param({30}, {10}, ("b", "a"), id="reversed-edges"),
        pytest.param({10}, {30}, ("a", "c"), id="missing-transition"),
    ],
)
def test_graph_to_nfa_rejects_invalid_paths(path_graph, starts, finals, word):
    nfa = graph_to_nfa(path_graph, starts, finals)

    assert not nfa.accepts(word)


@pytest.mark.parametrize(
    ("starts", "finals"),
    [
        pytest.param({10, 40}, {30}, id="unknown-start"),
        pytest.param({10}, {30, 40}, id="unknown-final"),
    ],
)
def test_graph_to_nfa_rejects_unknown_vertices(path_graph, starts, finals):
    with pytest.raises(ValueError):
        graph_to_nfa(path_graph, starts, finals)


def test_graph_to_nfa_only_starts_default_to_all_vertices(path_graph):
    nfa = graph_to_nfa(path_graph, set(), {30})

    assert nfa.start_states == {State(10), State(20), State(30)}
    assert nfa.final_states == {State(30)}


def test_graph_to_nfa_only_finals_default_to_all_vertices(path_graph):
    nfa = graph_to_nfa(path_graph, {10}, set())

    assert nfa.start_states == {State(10)}
    assert nfa.final_states == {State(10), State(20), State(30)}


@pytest.mark.parametrize(
    "word",
    [
        pytest.param(("a", "b"), id="first-branch"),
        pytest.param(("a", "c"), id="second-branch"),
    ],
)
def test_graph_to_nfa_preserves_both_targets_for_one_label(word):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 30, label="a")
    graph.add_edge(20, 40, label="b")
    graph.add_edge(30, 40, label="c")

    nfa = graph_to_nfa(graph, {10}, {40})

    assert nfa.accepts(word)


@pytest.mark.parametrize("label", ["a", "b"])
def test_graph_to_nfa_preserves_parallel_edge_labels(label):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(10, 20, label="b")

    nfa = graph_to_nfa(graph, {10}, {20})

    assert nfa.accepts([label])


def test_graph_to_nfa_preserves_self_loop():
    graph = MultiDiGraph()
    graph.add_edge(10, 10, label="a")

    nfa = graph_to_nfa(graph, {10}, {10})

    assert nfa.accepts(["a", "a"])


def test_graph_to_nfa_preserves_multicharacter_label():
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="label")

    nfa = graph_to_nfa(graph, {10}, {20})

    assert nfa.accepts(["label"])


@pytest.mark.parametrize(
    ("finals", "expected"),
    [
        pytest.param({10}, True, id="shared-start-and-final"),
        pytest.param({20}, False, id="disconnected-start-and-final"),
    ],
)
def test_graph_to_nfa_empty_word_on_isolated_vertices(finals, expected):
    graph = MultiDiGraph()
    graph.add_nodes_from([10, 20])

    nfa = graph_to_nfa(graph, {10}, finals)

    assert nfa.accepts([]) == expected


def test_graph_to_nfa_preserves_isolated_state(path_graph):
    path_graph.add_node(40)

    nfa = graph_to_nfa(path_graph, {10}, {30})

    assert State(40) in nfa.states


def test_graph_to_nfa_empty_graph_has_empty_language():
    nfa = graph_to_nfa(MultiDiGraph(), set(), set())

    assert nfa.is_empty()
