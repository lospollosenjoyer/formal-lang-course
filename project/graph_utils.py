from dataclasses import dataclass
from pathlib import Path

import cfpq_data
import pydot
from networkx import MultiDiGraph


@dataclass
class GraphInfo:
    """Node and edge counts and unique edge labels of a graph."""

    num_nodes: int
    num_edges: int
    labels: set[str]


def load_graph(graph_name: str) -> MultiDiGraph:
    """
    Load a graph from CFPQ_Data and return it.

    Args:
        graph_name: Name of the graph in the CFPQ_Data.

    Returns:
        MultiDiGraph: Graph loaded via name from CFPQ_Data.

    Raises:
        FileNotFoundError: The graph name is not in the CFPQ_Data dataset.
    """

    graph_path = cfpq_data.download(graph_name)
    return cfpq_data.graph_from_csv(graph_path)


def load_graph_info(graph_name: str) -> GraphInfo:
    """
    Load a graph from CFPQ_Data and return its basic information.

    Args:
        graph_name: Name of the graph in the CFPQ_Data.

    Returns:
        GraphInfo with node and edge counts and set of unique edge labels.

    Raises:
        FileNotFoundError: The graph name is not in the CFPQ_Data dataset.
    """
    graph = load_graph(graph_name)

    return GraphInfo(
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        labels={data["label"] for _, _, data in graph.edges(data=True)},
    )


def create_two_cycles_graph(
    num_first_cycle_nodes: int,
    first_cycle_label: str,
    num_second_cycle_nodes: int,
    second_cycle_label: str,
) -> MultiDiGraph:
    """
    Create a graph with two cycles sharing a node.

    Note:
        Cycle sizes exclude the shared node. A zero-sized cycle is represented by a self-loop.

    Args:
        num_first_cycle_nodes: Number of nodes in the first cycle, excluding the shared node.
        first_cycle_label: Label on edges of first cycle in the final graph.

        num_second_cycle_nodes: Number of nodes in the second cycle, excluding the shared node.
        second_cycle_label: Label on edges of second cycle in the final graph.

    Returns:
        MultiDiGraph: Final two cycles labeled graph.

    Raises:
        ValueError: Negative cycle size.
    """

    if num_first_cycle_nodes < 0 or num_second_cycle_nodes < 0:
        raise ValueError("Cycle sizes must be non-negative")

    if num_first_cycle_nodes == 0:
        graph = cfpq_data.labeled_cycle_graph(
            num_second_cycle_nodes + 1,
            label=second_cycle_label,
        )
        graph.add_edge(0, 0, label=first_cycle_label)
    elif num_second_cycle_nodes == 0:
        graph = cfpq_data.labeled_cycle_graph(
            num_first_cycle_nodes + 1,
            label=first_cycle_label,
        )
        graph.add_edge(0, 0, label=second_cycle_label)
    else:
        graph = cfpq_data.labeled_two_cycles_graph(
            num_first_cycle_nodes,
            num_second_cycle_nodes,
            labels=(first_cycle_label, second_cycle_label),
        )

    return graph


def save_two_cycles_graph(
    num_first_cycle_nodes: int,
    first_cycle_label: str,
    num_second_cycle_nodes: int,
    second_cycle_label: str,
    output_path: str | Path,
) -> None:
    """
    Create a graph with two cycles sharing a node and save it as DOT.

    Note:
        Cycle sizes exclude the shared node. A zero-sized cycle is represented by a self-loop.

    Args:
        num_first_cycle_nodes: Number of nodes in the first cycle, excluding the shared node.
        first_cycle_label: Label on edges of first cycle in the final graph.

        num_second_cycle_nodes: Number of nodes in the second cycle, excluding the shared node.
        second_cycle_label: Label on edges of second cycle in the final graph.

        output_path: Path to the output DOT file. An existing file is overwritten.

    Raises:
        ValueError: Negative cycle size.
    """

    graph = create_two_cycles_graph(
        num_first_cycle_nodes,
        first_cycle_label,
        num_second_cycle_nodes,
        second_cycle_label,
    )

    dot_graph = pydot.Dot(graph_type="digraph", strict=False)
    for node in graph.nodes:
        dot_graph.add_node(pydot.Node(str(node)))

    for source, target, data in graph.edges(data=True):
        # make_quoted escapes quotes but leaves backslashes unchanged.
        label = pydot.make_quoted(data["label"].replace("\\", "\\\\"))
        dot_graph.add_edge(pydot.Edge(str(source), str(target), label=label))

    dot_graph.write_raw(output_path, encoding="utf-8")
