from itertools import count

from pyformlang.cfg import CFG, Production, Terminal, Variable


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
