from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """
    Build a minimal DFA for a regular expression.

    Args:
        regex: Expression using | for union, a space or . for concatenation,
            * for repetition and epsilon for the empty word.

    Returns:
        DeterministicFiniteAutomaton: Minimal DFA accepting the expression's language.
    """
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """
    Convert graph vertices to NFA states and labeled edges to transitions.

    Args:
        graph: Directed multigraph with edge symbols in the label attribute.
        start_states: Initial vertices; an empty set means all graph vertices.
        final_states: Accepting vertices; an empty set independently means all vertices.

    Returns:
        NondeterministicFiniteAutomaton: NFA accepting labels of paths from
            initial to accepting vertices.

    Raises:
        ValueError: A specified initial or accepting vertex is absent from the graph.
    """
    nodes = set(graph.nodes)
    if not start_states.issubset(nodes) or not final_states.issubset(nodes):
        raise ValueError("Start and final states must be vertices of the graph")

    nfa = NondeterministicFiniteAutomaton(states=nodes)

    for vertex in graph:
        if not start_states or vertex in start_states:
            nfa.add_start_state(vertex)

        if not final_states or vertex in final_states:
            nfa.add_final_state(vertex)

        for source, target, data in graph.out_edges(vertex, data=True):
            nfa.add_transition(source, data["label"], target)

    return nfa
