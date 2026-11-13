from copy import deepcopy
from itertools import product

import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.grammar_utils import cfg_to_weak_normal_form


def _words(alphabet, max_length):
    for length in range(max_length + 1):
        yield from product(alphabet, repeat=length)


@pytest.fixture(
    params=[
        pytest.param(("", set()), id="empty-language"),
        pytest.param(("S -> $", {()}), id="epsilon-only"),
        pytest.param(("S -> a", {("a",)}), id="single-terminal"),
        pytest.param(("S -> S | a | $", {(), ("a",)}), id="productive-self-unit-rule"),
        pytest.param(
            ("S -> A\nA -> B\nB -> A | a | $", {(), ("a",)}),
            id="unit-cycle-with-epsilon",
        ),
        pytest.param(
            ("S -> A B\nA -> a | $\nB -> b | $", {(), ("a",), ("b",), ("a", "b")}),
            id="nullable-children",
        ),
        pytest.param(
            ("S -> a B c d\nB -> b | $", {("a", "b", "c", "d"), ("a", "c", "d")}),
            id="long-mixed-body",
        ),
        pytest.param(
            (
                "S -> a S b c | $",
                {(), ("a", "b", "c"), ("a", "a", "b", "c", "b", "c")},
            ),
            id="recursive-long-mixed-body",
        ),
        pytest.param(("S -> A\nA -> S", set()), id="nonproductive-unit-cycle"),
        pytest.param(
            ("S -> a | D E\nD -> b\nU -> c", {("a",)}),
            id="nonproductive-and-unreachable-rules",
        ),
        pytest.param(
            (
                "S -> a WCNF_0 b WCNF_1\nWCNF_0 -> b\nWCNF_1 -> a",
                {("a", "b", "b", "a")},
            ),
            id="fresh-variable-name-collisions",
        ),
    ],
)
def grammar_case(request):
    text, language = request.param
    return CFG.from_text(text), language


def test_weak_normal_form_has_only_allowed_productions(grammar_case):
    cfg, _ = grammar_case
    result = cfg_to_weak_normal_form(cfg)

    for production in result.productions:
        body = production.body
        assert (
            not body
            or len(body) == 1
            and isinstance(body[0], Terminal)
            or len(body) == 2
            and all(isinstance(symbol, Variable) for symbol in body)
        ), (cfg.to_text(), production)


def test_weak_normal_form_preserves_bounded_language(grammar_case):
    cfg, expected = grammar_case
    result = cfg_to_weak_normal_form(cfg)
    alphabet = sorted({terminal.value for terminal in cfg.terminals} | {"unknown"})
    max_length = max(map(len, expected), default=0) + 1
    actual = {word for word in _words(alphabet, max_length) if result.contains(word)}

    assert actual == expected, cfg.to_text()


def test_weak_normal_form_preserves_recursive_language():
    cfg = CFG.from_text("S -> a S b | $")
    result = cfg_to_weak_normal_form(cfg)

    actual = {word for word in _words("ab", 7) if result.contains(word)}

    assert actual == {tuple("a" * n + "b" * n) for n in range(4)}


@pytest.mark.parametrize(
    "word",
    [
        pytest.param(("WCNF_0", "a"), id="first-terminal"),
        pytest.param(("a", "WCNF_0"), id="second-terminal"),
        pytest.param(("WCNF_0", "WCNF_1", "WCNF_2"), id="long-rule"),
    ],
)
def test_weak_normal_form_fresh_variables_do_not_shadow_terminals(word):
    start = Variable("S")
    cfg = CFG(
        start_symbol=start,
        productions=[Production(start, [Terminal(value) for value in word])],
    )

    result = cfg_to_weak_normal_form(cfg)

    assert {variable.value for variable in result.variables}.isdisjoint(word), (
        word,
        result.to_text(),
    )
    actual = {
        candidate
        for candidate in _words(sorted(set(word) | {"other"}), len(word) + 1)
        if result.contains(candidate)
    }
    assert actual == {word}


def test_weak_normal_form_removes_rules_unreachable_after_unit_elimination():
    cfg = CFG.from_text("S -> A\nA -> a")

    result = cfg_to_weak_normal_form(cfg)

    assert set(result.productions) == {Production(Variable("S"), [Terminal("a")])}


def test_weak_normal_form_distinguishes_same_named_terminal_and_variable():
    start, variable = Variable("S"), Variable("X")
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(start, [variable, Terminal("a")]),
            Production(start, [Terminal("X"), Terminal("a")]),
            Production(variable, [Terminal("b")]),
        ],
    )
    result = cfg_to_weak_normal_form(cfg)

    variables = {
        variable: Variable(index) for index, variable in enumerate(result.variables)
    }
    recognizer = CFG(
        start_symbol=variables[result.start_symbol],
        productions=[
            Production(
                variables[production.head],
                [
                    variables[symbol] if isinstance(symbol, Variable) else symbol
                    for symbol in production.body
                ],
            )
            for production in result.productions
        ],
    )
    actual = {word for word in _words(["X", "a", "b"], 3) if recognizer.contains(word)}

    assert actual == {("b", "a"), ("X", "a")}


@pytest.mark.parametrize(
    ("start_value", "variable_value", "terminal_value"),
    [
        pytest.param("V0", "t0", "e", id="temporary-symbol-names"),
        pytest.param("C#CNF#1", "t0#CNF#", "WCNF_0", id="cnf-helper-names"),
        pytest.param("WCNF_0", "e", "t0", id="marker-and-start-names"),
    ],
)
def test_weak_normal_form_preserves_language_with_service_name_collisions(
    start_value, variable_value, terminal_value
):
    start, variable = Variable(start_value), Variable(variable_value)
    terminal = Terminal(terminal_value)
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(start, [terminal, variable, terminal]),
            Production(start, []),
            Production(variable, [Terminal(variable_value)]),
            Production(variable, []),
        ],
    )

    result = cfg_to_weak_normal_form(cfg)

    assert result.start_symbol == start
    alphabet = {symbol.value for symbol in cfg.terminals | result.terminals}
    assert {symbol.value for symbol in result.variables}.isdisjoint(alphabet)
    actual = {
        word
        for word in _words(sorted(alphabet | {"unknown"}), 4)
        if result.contains(word)
    }
    assert actual == {
        (),
        (terminal_value, terminal_value),
        (terminal_value, variable_value, terminal_value),
    }


def test_weak_normal_form_preserves_custom_start_symbol():
    cfg = CFG.from_text("Root -> A\nA -> label | $\nS -> other", start_symbol="Root")

    assert cfg_to_weak_normal_form(cfg).start_symbol == Variable("Root")


def test_weak_normal_form_does_not_modify_input():
    cfg = CFG.from_text("S -> A | a S b\nA -> $ | c")
    before = deepcopy(cfg)

    cfg_to_weak_normal_form(cfg)

    assert cfg.start_symbol == before.start_symbol
    assert cfg.variables == before.variables
    assert cfg.terminals == before.terminals
    assert cfg.productions == before.productions


def test_weak_normal_form_handles_grammar_without_start_symbol():
    result = cfg_to_weak_normal_form(CFG())

    assert result.start_symbol is None
    assert not result.productions


def test_weak_normal_form_does_not_infer_start_from_variable_with_none_value():
    cfg = CFG(productions=[Production(Variable(None), [Terminal("a")])])

    result = cfg_to_weak_normal_form(cfg)

    assert result.start_symbol is None
    assert not result.productions


def test_weak_normal_form_preserves_explicit_start_with_none_value():
    start = Variable(None)
    cfg = CFG(start_symbol=start, productions=[Production(start, [Terminal("a")])])

    result = cfg_to_weak_normal_form(cfg)

    assert result.start_symbol is start
    assert result.contains(["a"])
    assert not result.contains([])
