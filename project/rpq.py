from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automata_utils import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """
    Find vertex pairs connected by a path matching a regular expression.

    Note:
        Paths of length zero are included if the expression accepts the empty word.

    Args:
        regex: Regular expression in the format accepted by regex_to_dfa.
        graph: Directed multigraph with edge symbols in the label attribute.
        start_nodes: Initial vertices; an empty set means all graph vertices.
        final_nodes: Accepting vertices; an empty set independently means all vertices.

    Returns:
        set[tuple[int, int]]: Matching pairs of original graph vertex identifiers.

    Raises:
        ValueError: A specified initial or accepting vertex is absent from the graph.
    """
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes), set(final_nodes))
    )
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    product = intersect_automata(graph_fa, regex_fa)
    if not product.start_states or not product.final_states:
        return set()

    start_states = list(product.start_states)
    final_states = list(product.final_states)
    reachable = product.transitive_closure()[start_states][:, final_states].tocoo()

    result = set()
    for row, column in zip(reachable.row, reachable.col):
        source, _ = product.states[start_states[row]].value
        target, _ = product.states[final_states[column]].value
        result.add((source, target))

    return result


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """
    Find matching vertex pairs using multiple-source BFS over sparse matrices.

    For each query state, frontier and visited matrices have one row per source
    and one column per graph vertex. A BFS step multiplies a frontier by the
    graph matrix for a label and moves it to the corresponding query state.
    Visited entries are tracked separately for each source and query state.

    Args:
        regex: Regular expression in the format accepted by regex_to_dfa.
        graph: Directed multigraph with edge symbols in the label attribute.
        start_nodes: Initial vertices; an empty set means all graph vertices.
        final_nodes: Accepting vertices; an empty set independently means all vertices.

    Returns:
        set[tuple[int, int]]: Matching pairs of original graph vertex identifiers,
            including zero-length paths when the expression accepts the empty word.

    Raises:
        ValueError: A specified initial or accepting vertex is absent from the graph.
    """
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(graph, set(start_nodes), set(final_nodes))
    )
    query = regex_to_dfa(regex)
    if not graph_fa.start_states or not query.final_states:
        return set()

    sources = list(graph_fa.start_states)
    shape = (len(sources), graph_fa.num_states)
    initial = csr_matrix(
        ([True] * len(sources), (range(len(sources)), sources)),
        shape=shape,
        dtype=bool,
    )
    visited = {state: csr_matrix(shape, dtype=bool) for state in query.states}
    frontier = {state: initial for state in query.start_states}
    visited.update(frontier)
    transitions = query.to_dict()

    while frontier:
        candidates = {}
        for state, matrix in frontier.items():
            for symbol, target in transitions.get(state, {}).items():
                graph_matrix = graph_fa.matrices.get(symbol)
                if graph_matrix is None:
                    continue
                reached = matrix @ graph_matrix
                if target in candidates:
                    candidates[target] = candidates[target] + reached
                else:
                    candidates[target] = reached

        frontier = {}
        for state, matrix in candidates.items():
            new = matrix > visited[state]
            if new.nnz:
                frontier[state] = new
                visited[state] = visited[state] + new

    reachable = csr_matrix(shape, dtype=bool)
    for state in query.final_states:
        reachable = reachable + visited[state]
    finals = list(graph_fa.final_states)
    reachable = reachable[:, finals].tocoo()
    return {
        (graph_fa.states[sources[row]].value, graph_fa.states[finals[column]].value)
        for row, column in zip(reachable.row, reachable.col)
    }
