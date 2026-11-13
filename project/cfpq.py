from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import State, Symbol
from scipy.sparse import csr_matrix, eye

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.automata_utils import graph_to_nfa
from project.grammar_utils import cfg_to_weak_normal_form


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """
    Find vertex pairs connected by a path in a CFG's language using Hellings.

    Each discovered triple (variable, source, target) is processed once. For
    A -> B C, paths for B and C are joined at their common endpoint. Incoming
    and outgoing indices restrict these joins to matching variables and vertices.

    Args:
        cfg: Query grammar, not necessarily in weak Chomsky normal form.
        graph: Directed graph or multigraph with edge symbols in the label attribute.
        start_nodes: Initial vertices; None or an empty set means all graph vertices.
        final_nodes: Accepting vertices; None or an empty set independently means all.

    Returns:
        set[tuple[int, int]]: Matching pairs of original graph vertex identifiers,
            including zero-length paths when the grammar generates the empty word.

    Raises:
        ValueError: A specified initial or accepting vertex is absent from the graph.
    """
    automaton = graph_to_nfa(
        graph,
        set(start_nodes) if start_nodes is not None else set(),
        set(final_nodes) if final_nodes is not None else set(),
    )
    grammar = cfg_to_weak_normal_form(cfg)
    terminal_rules = defaultdict(set)
    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    epsilon_heads = set()
    for production in grammar.productions:
        if not production.body:
            epsilon_heads.add(production.head)
        elif len(production.body) == 1:
            terminal_rules[production.body[0].value].add(production.head)
        else:
            left, right = production.body
            left_rules[left].add((production.head, right))
            right_rules[right].add((production.head, left))

    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    queue = deque()

    def add_path(variable: Variable, source: State, target: State) -> None:
        if target not in outgoing[variable, source]:
            outgoing[variable, source].add(target)
            incoming[variable, target].add(source)
            queue.append((variable, source, target))

    for variable in epsilon_heads:
        for state in automaton.states:
            add_path(variable, state, state)
    for source, symbol, target in automaton:
        for variable in terminal_rules.get(symbol.value, ()):
            add_path(variable, source, target)

    while queue:
        variable, source, target = queue.popleft()
        for head, left in right_rules.get(variable, ()):
            for start in incoming.get((left, source), ()):
                add_path(head, start, target)
        for head, right in left_rules.get(variable, ()):
            for final in outgoing.get((right, target), ()):
                add_path(head, source, final)

    return {
        (source.value, target.value)
        for source in automaton.start_states
        for target in outgoing.get((grammar.start_symbol, source), ())
        if target in automaton.final_states
    }


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """
    Find vertex pairs connected by a path in a CFG's language using sparse matrices.

    Each grammar variable has a boolean reachability matrix. A rule A -> B C
    adds the product of B's and C's matrices to A's matrix. Rules are applied
    until no matrix gains an entry; epsilon rules initialize diagonal entries.

    Args:
        cfg: Query grammar, not necessarily in weak Chomsky normal form.
        graph: Directed graph or multigraph with edge symbols in the label attribute.
        start_nodes: Initial vertices; None or an empty set means all graph vertices.
        final_nodes: Accepting vertices; None or an empty set independently means all.

    Returns:
        set[tuple[int, int]]: Matching pairs of original graph vertex identifiers,
            including zero-length paths when the grammar generates the empty word.

    Raises:
        ValueError: A specified initial or accepting vertex is absent from the graph.
    """
    graph_fa = AdjacencyMatrixFA(
        graph_to_nfa(
            graph,
            set(start_nodes) if start_nodes is not None else set(),
            set(final_nodes) if final_nodes is not None else set(),
        )
    )
    if not graph_fa.start_states or not graph_fa.final_states:
        return set()

    grammar = cfg_to_weak_normal_form(cfg)
    shape = (graph_fa.num_states, graph_fa.num_states)
    matrices = {
        variable: csr_matrix(shape, dtype=bool) for variable in grammar.variables
    }
    identity = eye(graph_fa.num_states, format="csr", dtype=bool)
    binary_rules = []
    for production in grammar.productions:
        head = production.head
        if not production.body:
            matrices[head] = matrices[head] + identity
        elif len(production.body) == 1:
            symbol = Symbol(production.body[0].value)
            terminal_matrix = graph_fa.matrices.get(symbol)
            if terminal_matrix is not None:
                matrices[head] = matrices[head] + terminal_matrix
        else:
            left, right = production.body
            binary_rules.append((head, left, right))

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_rules:
            expanded = matrices[head] + matrices[left] @ matrices[right]
            if expanded.nnz > matrices[head].nnz:
                matrices[head] = expanded
                changed = True

    reachable = matrices.get(grammar.start_symbol)
    if reachable is None:
        return set()
    starts = list(graph_fa.start_states)
    finals = list(graph_fa.final_states)
    reachable = reachable[starts][:, finals].tocoo()
    return {
        (graph_fa.states[starts[row]].value, graph_fa.states[finals[column]].value)
        for row, column in zip(reachable.row, reachable.col)
    }
