import pygraphviz
from ltlf2dfa.parser.ltlf import LTLfParser
from pythomata import SymbolicAutomaton
from pythomata.simulator import AutomatonSimulator


def ltlf_to_dfa_graph(dfa_str: str) -> SymbolicAutomaton:
    """Converts an LTLf formula string to a DFA graph."""

    # Parse the DOT string
    graph = pygraphviz.AGraph(dfa_str)

    states = set()
    transitions = set()
    initial_state = None
    final_states = set()

    # Extract states (and final states)
    for node in graph.nodes():
        if node == "init":
            continue
        states.add(int(node))
        if node.attr["shape"] == "doublecircle":
            final_states.add(int(node))

    # Extract transitions (and initial state)
    for edge in graph.edges():
        label = edge.attr["label"]
        if edge[0] == "init":
            initial_state = int(edge[1])
            continue
        transitions.add((int(edge[0]), label, int(edge[1])))

    # Construct a SymbolicAutomaton
    automaton = SymbolicAutomaton()._from_transitions(
        states=states,
        initial_state=initial_state,
        final_states=final_states,
        transitions=transitions,
    )

    # Minimize and determinize the automaton (not required since LTLf2DFA already does this)
    # automaton = automaton.minimize()
    # automaton = automaton.determinize()

    return automaton


def ltlf_to_dfa(formula_str: str) -> AutomatonSimulator:
    """Converts an LTLf formula string into a working DFA."""
    parser = LTLfParser()
    formula = parser(formula_str)
    dfa_str = formula.to_dfa()
    return AutomatonSimulator(ltlf_to_dfa_graph(dfa_str))
