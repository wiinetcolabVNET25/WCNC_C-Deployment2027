import pandas as pd
import igraph as ig
import networkx as nx
import random
import numpy as np
from sklearn.metrics import adjusted_rand_score

class Graph:

    @staticmethod
    def create_cell_transition_graph(trace: pd.DataFrame) -> nx.DiGraph:
        """
        Create a directed weighted graph from vehicle cell transitions.

        Expects a trace with columns:
            - vehicle_id
            - x
            - y
            - cell_dwell_time

        ordered by vehicle_id.
        
        Nodes:
            Cells represented as (x, y) tuples.

        Edges:
            Directed transitions between consecutive cells visited by
            each vehicle.

        Edge attribute:
            weight = number of observed transitions.

        Expects trace columns:
            vehicle_id, x, y

        The trace must be ordered chronologically within each vehicle.
        """

        graph = nx.DiGraph()

        for _, vehicle_trace in trace.groupby("vehicle_id", sort=False):

            cells = list(zip(vehicle_trace["x"], vehicle_trace["y"]))

            for source, target in zip(cells[:-1], cells[1:]):

                # Do not count remaining in the same cell as a transition
                if source == target:
                    continue

                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += 1
                else:
                    graph.add_edge(source, target, weight=1)

        return graph

    @staticmethod
    def create_cell_transition_spatial_graph(trace: pd.DataFrame,
        flow_weight: float = 1.0, spatial_weight: float = 1.0) -> nx.DiGraph:
        """
        Create a directed weighted cell graph combining mobility
        transitions and spatial adjacency.

        Expects a trace with columns:
            - vehicle_id
            - x
            - y
            - cell_dwell_time

        ordered by vehicle_id.
        
        Nodes:
            Cells represented as (x, y) tuples.

        Edges:
            - Observed vehicle transitions between consecutive cells.
            - Spatial adjacency between cells sharing an edge.

        Edge weight:
            flow_weight * transition_count
            + spatial_weight * spatial_adjacency

        Spatial adjacency uses 4-neighbor connectivity:
            (x+1, y)
            (x-1, y)
            (x, y+1)
            (x, y-1)

        Parameters
        ----------
        trace:
            Trace dataframe containing x and y columns.

        flow_weight:
            Weight assigned to each observed transition.

        spatial_weight:
            Weight assigned to each pair of spatially adjacent cells.

        Returns
        -------
        nx.DiGraph
        """

        graph = nx.DiGraph()

        # --------------------------------------------------
        # Get all cells appearing in the trace
        # --------------------------------------------------

        cells = set(zip(trace["x"], trace["y"]))

        graph.add_nodes_from(cells)

        # --------------------------------------------------
        # Mobility transitions
        # --------------------------------------------------

        for _, vehicle_trace in trace.groupby("vehicle_id", sort=False):

            vehicle_cells = list(zip(vehicle_trace["x"], vehicle_trace["y"]))

            for source, target in zip(vehicle_cells[:-1], vehicle_cells[1:]):

                if source == target:
                    continue

                if graph.has_edge(source, target):
                    graph[source][target]["flow_weight"] += 1
                else:
                    graph.add_edge(source, target, flow_weight=1, spatial_weight=0)

        # --------------------------------------------------
        # Spatial adjacency
        # --------------------------------------------------

        for x, y in cells:

            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

            for neighbor in neighbors:

                if neighbor not in cells:
                    continue

                if graph.has_edge((x, y), neighbor):
                    graph[(x, y)][neighbor]["spatial_weight"] = 1
                else:
                    graph.add_edge((x, y), neighbor, flow_weight=0, spatial_weight=1)

        # --------------------------------------------------
        # Combine weights
        # --------------------------------------------------

        for source, target, data in graph.edges(data=True):
            data["weight"] = (
                flow_weight * data["flow_weight"]
                + spatial_weight * data["spatial_weight"]
            )

        return graph

    @staticmethod
    def create_cell_transition_probability_graph(trace: pd.DataFrame) -> nx.DiGraph:
        """
        Create a directed weighted graph where edge weights represent
        transition probabilities P(target | source).

        Nodes:
            Cells represented as (x, y) tuples.

        Edges:
            source -> target

        Edge weight:
            Probability of transitioning to `target` given that the
            vehicle leaves `source`.

        Expects trace columns:
            vehicle_id, x, y

        The trace must be ordered chronologically within each vehicle.
        """

        graph = nx.DiGraph()

        # First count transitions
        transition_counts = {}

        for _, vehicle_trace in trace.groupby("vehicle_id", sort=False):

            cells = list(zip(vehicle_trace["x"], vehicle_trace["y"]))

            for source, target in zip(cells[:-1], cells[1:]):

                if source == target:
                    continue

                transition_counts[(source, target)] = transition_counts.get((source, target), 0) + 1

        # Add nodes
        for cell in set(
            cell
            for source, target in transition_counts
            for cell in (source, target)):
            graph.add_node(cell)

        # Convert counts to conditional probabilities
        outgoing_totals = {}

        for (source, target), count in transition_counts.items():
            outgoing_totals[source] = outgoing_totals.get(source, 0) + count

        for (source, target), count in transition_counts.items():
            probability = count / outgoing_totals[source]

            graph.add_edge(source, target, weight=probability, count=count)

        return graph

    @staticmethod
    def detect_leiden_communities(graph: nx.DiGraph,
        resolution: float = 1.0, objective_function: str = "modularity",
        beta: float = 0.01, n_iterations: int = 2,
        node_weights=None, node_in_weights=None,
        initial_membership=None) -> tuple[dict, float, int]:
        """
        Detect communities in a weighted directed graph using Leiden.

        Parameters
        ----------
        graph:
            Weighted NetworkX graph.

        resolution:
            Resolution parameter controlling community granularity.
            Higher values generally produce more/smaller communities.

        objective_function:
            Leiden objective function. Common choices are:
                - "modularity"
                - "CPM"

        beta:
            Randomness parameter used by the Leiden refinement phase.

        n_iterations:
            Number of Leiden iterations.

        node_weights:
            Optional node weights.

        node_in_weights:
            Optional incoming node weights for directed graphs.

        initial_membership:
            Optional initial community assignment.

        Returns
        -------
        communities:
            Dictionary mapping NetworkX nodes to community IDs.

        quality:
            Quality/modularity value returned by Leiden.

        n_communities:
            Number of discovered communities.
        """


        nodes = list(graph.nodes())

        node_to_id = { node: i for i, node in enumerate(nodes) }

        edges = [
            (node_to_id[source], node_to_id[target])
            for source, target in graph.edges()
        ]

        weights = [
            graph[source][target].get("weight", 1.0)
            for source, target in graph.edges()
        ]

        g = ig.Graph(n=len(nodes), edges=edges, directed=graph.is_directed())

        g.es["weight"] = weights

        partition = g.community_leiden(
            objective_function=objective_function,
            weights="weight",
            resolution=resolution,
            beta=beta,
            initial_membership=initial_membership,
            n_iterations=n_iterations,
            node_weights=node_weights,
            node_in_weights=node_in_weights,
        )

        communities = {
            node: community_id
            for node, community_id in zip(nodes, partition.membership)
        }

        n_communities = len(set(partition.membership))

        return (communities, partition.quality, n_communities)

    @staticmethod
    def evaluate_leiden(graph: nx.DiGraph,
        resolutions: list[float] = (0.5, 1.0, 1.5, 2.0),
        objective_function: str = "modularity",
        beta: float = 0.01, n_iterations: int = 2,
        n_runs: int = 10) -> pd.DataFrame:
        """
        Runs leiden algorithm for multiple resolution values n_runs each.

        Updates random.seed() every run with the run's id (0, 1, 2, ...).

        Returns
        -------

        resolution, run, objective_function, beta, n_iterations, n_communities, quality for each run
        """

        results = []

        for resolution in resolutions:
            for run in range(n_runs):

                random.seed(run)

                communities, quality, n_communities = (
                    Graph.detect_leiden_communities(
                        graph=graph,
                        resolution=resolution,
                        objective_function=objective_function,
                        beta=beta, n_iterations=n_iterations))

                results.append({
                    "resolution": resolution,
                    "run": run,
                    "objective_function": objective_function,
                    "beta": beta,
                    "n_iterations": n_iterations,
                    "n_communities": n_communities,
                    "quality": quality,
                })

        return pd.DataFrame(results)

    @staticmethod
    def build_cooccurrence_matrix(partitions: list[dict], nodes: list) -> np.ndarray:
        """
        Build a co-occurrence matrix from graph partitions.

        Entry (i, j) is the fraction of partitions in which
        nodes i and j belong to the same community.
        """

        n = len(nodes)
        matrix = np.zeros((n, n), dtype=float)

        for partition in partitions:

            labels = np.array([partition[node] for node in nodes])

            matrix += (labels[:, None] == labels[None, :])

        matrix /= len(partitions)

        return matrix

    @staticmethod
    def cooccurrence_to_graph(matrix: np.ndarray, nodes: list, threshold: float = 0.5) -> nx.Graph:

        graph = nx.Graph()

        graph.add_nodes_from(nodes)

        n = len(nodes)

        for i in range(n):
            for j in range(i + 1, n):

                weight = matrix[i, j]

                if weight >= threshold:
                    graph.add_edge(nodes[i], nodes[j], weight=float(weight))

        return graph

    @staticmethod
    def repair_isolated_vertices(consensus: np.ndarray, thresholded: np.ndarray) -> np.ndarray:
        """
        Restore edges for vertices that became isolated after
        thresholding the consensus matrix.

        For each isolated vertex, restore its strongest edge
        from the original (unthresholded) consensus matrix.
        """
        repaired = thresholded.copy()

        row_sums = repaired.sum(axis=1) - np.diag(repaired)
        isolated = np.where(row_sums == 0)[0]

        for i in isolated:
            # Find strongest connection in the original consensus matrix.
            weights = consensus[i].copy()
            weights[i] = -np.inf

            j = np.argmax(weights)

            repaired[i, j] = consensus[i, j]
            repaired[j, i] = consensus[j, i]

        np.fill_diagonal(repaired, 1.0)

        return repaired 

    @staticmethod
    def leiden_consensus(graph: nx.Graph, resolution: float = 1.0,
        n_runs: int = 50, threshold: float = 0.6, beta: float = 0.01,
        n_iterations: int = 2, max_consensus_iterations: int = 100, seed: int = 42):
        """
        Recursive consensus clustering using Leiden.

        Parameters
        ----------
        graph:
            Initial network.

        resolution:
            Leiden resolution parameter.

        n_runs:
            Number of Leiden partitions generated at each
            consensus iteration.

        threshold:
            Consensus pruning threshold.

        beta:
            Leiden refinement randomness parameter.

        n_iterations:
            Number of Leiden optimization iterations.

        max_consensus_iterations:
            Safety limit for recursive consensus iterations.

        seed:
            Base random seed.

        Returns
        -------
        dict
            Consensus partition and convergence information.
        """

        current_graph = graph.copy()
        history = []

        for consensus_iteration in range(max_consensus_iterations):

            print(f"Consensus iter {consensus_iteration}")

            nodes = list(current_graph.nodes())

            partitions = []

            # --------------------------------------------------
            # 1. Run Leiden n_runs times
            # --------------------------------------------------

            for run in range(n_runs):

                run_seed = seed + run

                # igraph uses Python's random module
                random.seed(run_seed)

                communities, quality, n_communities = (
                    Graph.detect_leiden_communities(
                        graph=current_graph,
                        resolution=resolution,
                        beta=beta,
                        n_iterations=n_iterations,
                    )
                )

                partitions.append(communities)

            # --------------------------------------------------
            # 2. Build consensus matrix
            # --------------------------------------------------

            consensus = Graph.build_cooccurrence_matrix(
                partitions=partitions,
                nodes=nodes,
            )

            # Keep the original, unthresholded matrix.
            original_consensus = consensus.copy()

            # --------------------------------------------------
            # 3. Check whether all partitions are identical
            # --------------------------------------------------

            reference = [
                partitions[0][node]
                for node in nodes
            ]

            all_equal = all(
                adjusted_rand_score(
                    reference,
                    [partition[node] for node in nodes],
                ) == 1.0
                for partition in partitions[1:]
            )

            # --------------------------------------------------
            # 4. Prune edges below threshold
            # --------------------------------------------------

            consensus[consensus < threshold] = 0.0

            np.fill_diagonal(consensus, 1.0)

            # --------------------------------------------------
            # 5. Repair isolated vertices
            # --------------------------------------------------

            consensus = Graph.repair_isolated_vertices(
                consensus=original_consensus,
                thresholded=consensus,
            )

            # --------------------------------------------------
            # Store iteration information
            # --------------------------------------------------

            history.append({
                "iteration": consensus_iteration,
                "n_communities": [
                    len(set(partition.values()))
                    for partition in partitions
                ],
                "all_equal": all_equal,
                "consensus_matrix": consensus.copy(),
            })

            # --------------------------------------------------
            # 6. Stop if partitions are identical
            # --------------------------------------------------

            if all_equal:
                return {
                    "partition": Graph.order_partition_by_size(partitions[0]),
                    "matrix": consensus,
                    "graph": current_graph,
                    "iterations": consensus_iteration + 1,
                    "converged": True,
                    "history": history,
                }

            # --------------------------------------------------
            # 7. Construct next consensus network
            # --------------------------------------------------

            current_graph = Graph.cooccurrence_to_graph(
                matrix=consensus,
                nodes=nodes,
                threshold=0.0,
            )

        # ------------------------------------------------------
        # Did not converge within maximum iterations
        # ------------------------------------------------------

        return {
            "partition": Graph.order_partition_by_size(partitions[0]),
            "matrix": consensus,
            "graph": current_graph,
            "iterations": max_consensus_iterations,
            "converged": False,
            "history": history,
        }

    @staticmethod
    def order_partition_by_size(partition: dict) -> dict:

        sizes = pd.Series(partition).value_counts()

        ordered_clusters = sizes.sort_values().index

        cluster_mapping = {
            old_id: new_id
            for new_id, old_id in enumerate(ordered_clusters)
        }

        return { node: cluster_mapping[cluster] for node, cluster in partition.items() }

    @staticmethod
    def assign_graph_communities_to_celldf(celldf: pd.DataFrame, communities: dict) \
        -> pd.DataFrame:

        result = celldf.copy()

        def get_community(cell_id):
            x, y = map(int, cell_id.split(":"))
            return communities.get((x, y), -1)

        result["cluster"] = result.index.map(get_community)

        return result