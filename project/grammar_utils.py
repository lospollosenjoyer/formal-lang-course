import re
from collections import defaultdict
from itertools import count

from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton

from project.automata_utils import regex_to_dfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """
    Convert a CFG to weak Chomsky normal form without changing its language.

    The result contains only A -> B C, A -> terminal and A -> epsilon rules.
    Symbols are renamed before pyformlang's CNF conversion to avoid collisions.
    A temporary terminal preserves epsilon rules through the conversion.

    Args:
        cfg: Grammar to convert. Its productions and start symbol are not modified.

    Returns:
        CFG: Equivalent grammar with the same start symbol, including the empty word
            if it belongs to the original language.
    """
    variables = {
        variable: Variable(f"V{index}") for index, variable in enumerate(cfg.variables)
    }
    terminals = {
        terminal: Terminal(f"t{index}") for index, terminal in enumerate(cfg.terminals)
    }
    epsilon_marker = Terminal("e")
    grammar = CFG(
        start_symbol=(
            variables[cfg.start_symbol] if cfg.start_symbol is not None else None
        ),
        productions={
            Production(
                variables[production.head],
                [
                    variables[symbol]
                    if isinstance(symbol, Variable)
                    else terminals[symbol]
                    for symbol in production.body
                ]
                or [epsilon_marker],
            )
            for production in cfg.productions
        },
    )
    grammar = (
        grammar.remove_useless_symbols()
        .eliminate_unit_productions()
        .remove_useless_symbols()
        .to_normal_form()
    )

    reserved_values = {variable.value for variable in cfg.variables}
    reserved_values.update(terminal.value for terminal in cfg.terminals)
    names = (f"WCNF_{index}" for index in count())
    fresh_variables = (Variable(name) for name in names if name not in reserved_values)
    restored = {renamed: terminal for terminal, renamed in terminals.items()}
    for variable in grammar.variables:
        restored[variable] = (
            cfg.start_symbol
            if variable == grammar.start_symbol
            else next(fresh_variables)
        )

    return CFG(
        start_symbol=cfg.start_symbol,
        productions={
            Production(
                restored[production.head],
                [
                    restored[symbol]
                    for symbol in production.body
                    if symbol != epsilon_marker
                ],
            )
            for production in grammar.productions
        },
    )


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """
    Build one RSM box per CFG variable, preserving the CFG's start variable.

    Each production is a path from its box's initial state to an accepting
    state. A variable in the body becomes a call to the corresponding box.
    An empty production marks the initial state as accepting. If a variable
    and a terminal share a name, the box gets a fresh label.
    """
    start = cfg.start_symbol
    if start is None:
        return RecursiveAutomaton()

    terminal_values = {terminal.value for terminal in cfg.terminals}
    reserved_values = terminal_values | {variable.value for variable in cfg.variables}
    names = (f"RSM_{index}" for index in count())
    fresh_names = (name for name in names if name not in reserved_values)
    labels = {
        variable: Symbol(
            next(fresh_names) if variable.value in terminal_values else variable.value
        )
        for variable in cfg.variables
    }
    automata = {
        variable: NondeterministicFiniteAutomaton(start_state={0})
        for variable in cfg.variables
    }
    states = count(1)
    for production in cfg.productions:
        automaton = automata[production.head]
        source = 0
        for symbol in production.body:
            if isinstance(symbol, Epsilon):
                continue
            label = (
                labels[symbol] if isinstance(symbol, Variable) else Symbol(symbol.value)
            )
            target = next(states)
            automaton.add_transition(source, label, target)
            source = target
        automaton.add_final_state(source)

    start_box = Box(automata.pop(start), labels[start])
    rsm = RecursiveAutomaton(initial_label=start_box.label, boxes={start_box})
    
    for variable, automaton in automata.items():
        rsm.add_box(Box(automaton, labels[variable]))
    return rsm


_EBNF_TOKEN = re.compile(r"(?:\\.|[^\\()|+.*$ ])+|[\\()|+.*$]")


def _balanced_regex(parts: list[str], operator: str) -> str:
    """Keep the regex tree shallow when folding a sequence or alternatives."""
    while len(parts) > 1:
        pairs = [
            f"({parts[index]}{operator}{parts[index + 1]})"
            for index in range(0, len(parts) - 1, 2)
        ]
        if len(parts) % 2:
            pairs.append(parts[-1])
        parts = pairs
    return parts[0]


def _encode_ebnf_body(body: str, tokens: dict[str, str]) -> str:
    """Validate an RHS and encode each literal as a single regex token."""
    body = body.lstrip()
    trimmed = body.rstrip()
    if not trimmed:
        return "epsilon"
    
    escaped_tail = (len(trimmed) - len(trimmed.rstrip("\\"))) % 2
    body = body[: len(trimmed) + escaped_tail]

    alternatives, sequence = [], []
    groups = []
    needs_term = True
    for match in _EBNF_TOKEN.finditer(body):
        token = match.group()
        if token == "(":
            groups.append((alternatives, sequence))
            alternatives, sequence = [], []
            needs_term = True
        elif token == ")":
            if not groups or needs_term:
                raise ValueError("unexpected closing parenthesis")
            alternatives.append(_balanced_regex(sequence, "."))
            group = _balanced_regex(alternatives, "|")
            alternatives, sequence = groups.pop()
            sequence.append(group)
            needs_term = False
        elif token in {"|", "+", ".", "*"}:
            if needs_term:
                raise ValueError(f"expected a term before {token!r}")
            if token == "*":
                if not sequence[-1].endswith("*"):
                    sequence[-1] += "*"
                continue
            if token != ".":
                alternatives.append(_balanced_regex(sequence, "."))
                sequence = []
            needs_term = True
        else:
            if token == "\\":
                raise ValueError("trailing backslash")
            if token in {"epsilon", "$"}:
                sequence.append("epsilon")
            else:
                literal = re.sub(r"\\(.)", r"\1", token)
                sequence.append(tokens.setdefault(literal, f"t{len(tokens)}"))
            needs_term = False

    if groups or needs_term:
        raise ValueError("unclosed parenthesis or missing term")
    alternatives.append(_balanced_regex(sequence, "."))
    return _balanced_regex(alternatives, "|")


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    r"""
    Build an RSM from one `head -> expression` rule per nonblank line.

    A head is a nonempty name with surrounding whitespace ignored. Repeated
    heads are alternatives. Expressions use `|` or `+` for union, adjacency,
    spaces or `.` for concatenation, `*` for repetition and parentheses for
    grouping. An empty body, `epsilon` or `$` denotes the empty word.
    Backslash quotes the next character, so `a\.b` is one label `a.b` and
    `\\` is a literal backslash. Other characters form multi-character labels.
    A label naming a box is a call to that box. The start box is `S` if
    defined, otherwise the head of the first rule.

    Raises:
        ValueError: A nonblank line is not a rule or has an invalid expression.
    """
    rules = defaultdict(list)
    tokens: dict[str, str] = {}
    for number, line in enumerate(ebnf.splitlines(), start=1):
        if not line.strip():
            continue
        head, arrow, body = line.partition("->")
        head = head.strip()
        if not arrow or not head:
            raise ValueError(f"Line {number}: expected 'head -> expression'")
        try:
            rules[head].append(_encode_ebnf_body(body, tokens))
        except ValueError as error:
            raise ValueError(f"Line {number}: {error}") from error

    if not rules:
        return RecursiveAutomaton()

    symbols = {token: Symbol(literal) for literal, token in tokens.items()}
    boxes = set()
    for head, alternatives in rules.items():
        dfa = regex_to_dfa(_balanced_regex(alternatives, "|"))
        automaton = NondeterministicFiniteAutomaton(
            states=dfa.states,
            start_state=dfa.start_states,
            final_states=dfa.final_states,
        )
        for source, symbol, target in dfa:
            automaton.add_transition(source, symbols[symbol.value], target)
        boxes.add(Box(automaton, Symbol(head)))

    start = "S" if "S" in rules else next(iter(rules))
    return RecursiveAutomaton(initial_label=Symbol(start), boxes=boxes)
