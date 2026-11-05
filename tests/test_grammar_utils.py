from copy import deepcopy
from itertools import product

import pytest
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from pyformlang.finite_automaton import Symbol

from project.grammar_utils import cfg_to_rsm, cfg_to_weak_normal_form, ebnf_to_rsm


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


def test_cfg_to_rsm_keeps_boxes_with_colliding_label_hashes():
    start, other = Variable(-1), Variable(-2)
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(variable, [Terminal(value)])
            for variable in (start, other)
            for value in ("a", 0)
        ],
    )

    result = cfg_to_rsm(cfg)

    assert set(result.boxes) == {Symbol(-1), Symbol(-2)}


def test_cfg_to_rsm_renames_calls_without_shadowing_terminals_or_boxes():
    start, call, existing = Variable("S"), Variable("A"), Variable("RSM_1")
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(start, [call, Terminal("b")]),
            Production(start, [Terminal("A")]),
            Production(start, [Terminal("RSM_0")]),
            Production(call, [Terminal("a")]),
            Production(existing, [Terminal("c")]),
        ],
    )

    result = cfg_to_rsm(cfg)

    (call_label,) = set(result.boxes) - {Symbol("S"), Symbol("RSM_1")}
    assert Symbol("RSM_1") in result.boxes
    assert call_label not in {Symbol("A"), Symbol("RSM_0")}
    assert result.get_box(call_label).dfa.accepts(("a",))
    start_dfa = result.get_box(result.initial_label).dfa
    assert all(
        start_dfa.accepts(word) for word in ((call_label, "b"), ("A",), ("RSM_0",))
    )


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        pytest.param(Epsilon(), ("a", "b"), id="explicit-epsilon"),
        pytest.param(
            Terminal("epsilon"), ("a", Symbol("epsilon"), "b"), id="literal-epsilon"
        ),
    ],
)
def test_cfg_to_rsm_distinguishes_epsilon_from_literal(symbol, expected):
    start = Variable("S")
    cfg = CFG(
        start_symbol=start,
        productions=[
            Production(start, [Terminal("a"), symbol, Terminal("b")], filtering=False)
        ],
    )

    result = cfg_to_rsm(cfg)

    start_dfa = result.get_box(result.initial_label).dfa
    words = [("a", "b"), ("a", Symbol("epsilon"), "b")]
    assert {word for word in words if start_dfa.accepts(word)} == {expected}


def test_cfg_to_rsm_does_not_infer_start_from_none_valued_variable():
    cfg = CFG(productions=[Production(Variable(None), [Terminal("a")])])

    assert not cfg_to_rsm(cfg).boxes


def test_cfg_to_rsm_preserves_explicit_none_valued_start():
    start = Variable(None)
    cfg = CFG(start_symbol=start, productions=[Production(start, [Terminal("a")])])

    assert cfg_to_rsm(cfg).initial_label == Symbol(None)


@pytest.mark.parametrize(
    ("grammar", "start"),
    [
        pytest.param("Root -> a\nS -> b", "S", id="late-s"),
        pytest.param("Root -> a\nOther -> b", "Root", id="first-head"),
    ],
)
def test_ebnf_to_rsm_selects_initial_box(grammar, start):
    assert ebnf_to_rsm(grammar).initial_label == Symbol(start)


def test_ebnf_to_rsm_empty_body_accepts_epsilon():
    result = ebnf_to_rsm("S ->   ")

    assert result.get_box(result.initial_label).dfa.accepts(())


def test_ebnf_to_rsm_without_rules_has_no_boxes():
    assert not ebnf_to_rsm("\n\t\n").boxes


def test_ebnf_to_rsm_combines_repeated_heads_as_alternatives():
    result = ebnf_to_rsm("S -> a\nS -> b")

    start_dfa = result.get_box(result.initial_label).dfa
    assert {word for word in _words("ab", 2) if start_dfa.accepts(word)} == {
        ("a",),
        ("b",),
    }


def test_ebnf_to_rsm_preserves_calls_to_other_boxes():
    result = ebnf_to_rsm("S -> a A b\nA -> c")

    called_box = result.get_box(Symbol("A"))
    assert called_box is not None
    assert called_box.dfa.accepts(("c",))
    start_dfa = result.get_box(result.initial_label).dfa
    assert start_dfa.accepts(("a", called_box.label, "b"))


def test_ebnf_to_rsm_accepts_200_alternatives_for_one_head():
    symbols = tuple(f"a{index}" for index in range(200))
    result = ebnf_to_rsm("\n".join(f"S -> {symbol}" for symbol in symbols))

    start_dfa = result.get_box(result.initial_label).dfa
    assert all(start_dfa.accepts((symbol,)) for symbol in symbols)
    assert not start_dfa.accepts(())
    assert not start_dfa.accepts((symbols[0], symbols[-1]))


def test_ebnf_to_rsm_accepts_200_symbols_in_one_body():
    word = tuple(f"a{index}" for index in range(200))
    result = ebnf_to_rsm("S -> " + " ".join(word))

    start_dfa = result.get_box(result.initial_label).dfa
    assert start_dfa.accepts(word)
    assert not start_dfa.accepts(word[:-1])
    assert not start_dfa.accepts(word[::-1])


def test_ebnf_to_rsm_preserves_a_tab_inside_a_literal_label():
    result = ebnf_to_rsm("S -> a\tb")

    start_dfa = result.get_box(result.initial_label).dfa
    assert start_dfa.accepts(("a\tb",))
    assert not start_dfa.accepts(("a", "b"))


@pytest.mark.parametrize(
    "invalid_line",
    ["A a", "-> a", "A -> )", "A -> (a", "A -> a |", "A -> a\\"],
)
def test_ebnf_to_rsm_rejects_malformed_nonblank_rules(invalid_line):
    with pytest.raises(ValueError, match=r"(?i)line 2\b"):
        ebnf_to_rsm(f"S -> a\n{invalid_line}\nA -> b")
