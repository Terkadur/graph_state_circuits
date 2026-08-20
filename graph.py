import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from random import random, shuffle, choice
import stim
import numpy as np


class Graph:

    def __init__(self, n_photons: int = 0, n_emitters: int = 0, n_memories: int = 0, edges: set[tuple[int, int]] = set()) -> None: # TODO: is_isomorphic
        self.n_photons = n_photons
        self.n_emitters = n_emitters
        self.n_memories = n_memories

        edges = [tuple(sorted(edge)) for edge in edges]
    
        for edge in edges:
            if edge[0] not in self.qubit_list:
                raise Exception(f"Unexpected qubit: {edge[0]}")

            if edge[1] not in self.qubit_list:
                raise Exception(f"Unexpected qubit: {edge[1]}")

            if edge[0] == edge[1]:
                raise Exception(f"Unexpected self-loop: {edge}")
            
        self.network = nx.Graph()
        self.network.add_nodes_from(self.qubit_list)
        self.network.add_edges_from(edges)


    @property
    def n_qubits(self):
        return self.n_photons + self.n_memories + self.n_emitters
    
    @property
    def photon_list(self):
        return list(range(self.n_photons))
    
    @property
    def memory_list(self):
        return list(range(self.n_photons, self.n_photons + self.n_memories))
    
    @property
    def emitter_list(self):
        return list(range(self.n_photons + self.n_memories, self.n_photons + self.n_memories + self.n_emitters))
    
    @property
    def qubit_list(self):
        return list(range(self.n_photons + self.n_memories + self.n_emitters))

    @property
    def edges(self):
        return [tuple(sorted(edge)) for edge in self.network.edges]

    @edges.setter
    def edges(self, new_edges):
        self.network = nx.Graph()
        self.network.add_nodes_from(self.qubit_list)
        self.network.add_edges_from(new_edges)

    def __repr__(self) -> str:
        return f"Graph(n_photons={self.n_photons}, n_memories={self.n_memories}, n_emitters={self.n_emitters}, edges={self.edges})"

    def __hash__(self) -> int:
        result = 0
        result ^= hash(self.n_photons)
        result ^= hash(self.n_memories)
        result ^= hash(self.n_emitters)
        result ^= hash(self.network)

        return result

    def __eq__(self, other) -> bool:
        if self.n_photons != other.n_photons:
            return False
        
        if self.n_memories != other.n_memories:
            return False
        
        if self.n_emitters != other.n_emitters:
            return False
        
        if self.network.adj != other.network.adj:
            return False
        
        return True

    def isomorphic(self, other) -> bool:
        return nx.is_isomorphic(self.network, other.network)

    def copy(self):
        return Graph(n_photons=self.n_photons,
                     n_emitters=self.n_emitters,
                     n_memories=self.n_memories,
                     edges=self.edges)

    def draw(self, labels: bool = True, legend: bool = True) -> None:
        pos = nx.nx_pydot.graphviz_layout(self.network, prog='neato')

        nx.draw_networkx_nodes(self.network, pos,
                               nodelist=self.photon_list,
                               node_color='#6494ED',
                               node_shape='o')

        nx.draw_networkx_nodes(self.network, pos,
                               nodelist=self.emitter_list,
                               node_color='#ED7964',
                               node_shape='s')

        nx.draw_networkx_nodes(self.network, pos,
                               nodelist=self.memory_list,
                               node_color='#C6E41B',
                               node_shape='^')

        nx.draw_networkx_edges(self.network, pos, width=1)

        if labels:
            label_pos = {k: [v[0], v[1]] for k, v in pos.items()}
            for memory in self.memory_list:
                label_pos[memory][1] -= 2
            nx.draw_networkx_labels(self.network, label_pos)

        if legend:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Photon',
                    markerfacecolor='#6494ED', markersize=10),
                Line2D([0], [0], marker='s', color='w', label='Emitter',
                    markerfacecolor='#ED7964', markersize=10),
                Line2D([0], [0], marker='^', color='w', label='Memory',
                    markerfacecolor='#C6E41B', markersize=10)
            ]

            plt.legend(handles=legend_elements)

        plt.show()

    def randomize(self, connected: bool = True, edge_prob: float = 0.5) -> None:
        self.network.remove_edges_from(self.edges)

        if connected:
            shuffled_qubits = self.qubit_list
            shuffle(shuffled_qubits)
            
            for qubit1 in self.qubit_list[1:]:
                qubit2 = choice(shuffled_qubits[:qubit1])
                self.network.add_edge(shuffled_qubits[qubit1], qubit2)
                
            for qubit1 in self.qubit_list:
                for qubit2 in range(qubit1+1, self.n_qubits):
                    if not self.network.has_edge(qubit1, qubit2) and random() < edge_prob:
                        self.network.add_edge(qubit1, qubit2)
                        
        else:
            for qubit1 in self.qubit_list:
                for qubit2 in range(qubit1 + 1, self.n_qubits):
                    if random() < edge_prob:
                        self.network.add_edge(qubit1, qubit2)

    def is_neighbor(self, qubit1: int, qubit2: int) -> bool:
        return self.network.has_edge(qubit1, qubit2)

    def add_edge(self, qubit1: int, qubit2: int):
        self.network.add_edge(min(qubit1, qubit2), max(qubit1, qubit2))

    def remove_edge(self, qubit1: int, qubit2: int):
        self.network.remove_edge(qubit1, qubit2)

    def get_nbhd(self, qubit: int) -> set[int]:
        return set(self.network.neighbors(qubit))

    def to_stim(self) -> stim.Circuit:
        circ_str = ""

        for qubit in self.qubit_list:
            circ_str = circ_str + "H " + str(qubit) + "\n"

        for edge in self.edges:
            circ_str = circ_str + "CZ " + \
                str(edge[0]) + " " + str(edge[1]) + "\n"

        return stim.Circuit(circ_str)

    def loc_comp(self, qubit: int):
        nbhd = self.get_nbhd(qubit)
        for neighbor1 in nbhd:
            for neighbor2 in nbhd:
                if neighbor1 <= neighbor2:
                    continue

                if self.is_neighbor(neighbor1, neighbor2):
                    self.remove_edge(neighbor1, neighbor2)
                else:
                    self.add_edge(neighbor1, neighbor2)

    def lc_path(self, other_graph, max_depth: int = np.inf, allow_isomorphic: bool = False) -> list[int]:
        if self == other_graph:
            return []
        if allow_isomorphic and self.isomorphic(other_graph):
            return []
        
        pending_graphs = {self}
        graph_to_path = {self: []}
        while len(pending_graphs) > 0:
            current_graph = pending_graphs.pop()
            if len(graph_to_path[current_graph]) > max_depth:
                continue 

            for qubit in current_graph.qubit_list:
                new_graph = current_graph.copy()
                new_graph.loc_comp(qubit)
                new_path = graph_to_path[current_graph] + [qubit]
                
                if allow_isomorphic:
                    isomorphic_graph = None
                    for old_graph in graph_to_path:
                        if new_graph.isomorphic(old_graph):
                            isomorphic_graph = old_graph
                            break
                    
                    if isomorphic_graph is not None:
                        if len(new_path) < len(graph_to_path[isomorphic_graph]):
                            graph_to_path[new_graph] = new_path
                    else:
                        pending_graphs.add(new_graph)
                        graph_to_path[new_graph] = new_path
                else:
                    if new_graph not in graph_to_path:
                        pending_graphs.add(new_graph)
                        graph_to_path[new_graph] = new_path
                    elif len(new_path) < len(graph_to_path[new_graph]):
                        graph_to_path[new_graph] = new_path

        if allow_isomorphic:
            for graph in graph_to_path:
                if other_graph.isomorphic(graph):
                    return graph_to_path[graph]
        else:
            if other_graph in graph_to_path:
                return graph_to_path[other_graph]

        return None

    def lc_equivalent(self, other_graph, allow_isomorphic: bool = False) -> bool:
        if self == other_graph:
            return True
        if allow_isomorphic and self.isomorphic(other_graph):
            return True
        
        pending_graphs = {self}
        checked_graphs = set()
        while len(pending_graphs) > 0:
            current_graph = pending_graphs.pop()
            checked_graphs.add(current_graph)

            for qubit in current_graph.qubit_list:
                if len(current_graph.get_nbhd(qubit)) == 0:
                    continue

                new_graph = current_graph.copy()
                new_graph.loc_comp(qubit)

                if new_graph == other_graph:
                    return True
                if allow_isomorphic and new_graph.isomorphic(other_graph):
                    return True
                
                if allow_isomorphic:
                    is_present = False
                    for graph in checked_graphs:
                        if new_graph.isomorphic(graph):
                            is_present = True
                            break
                    if not is_present:
                        pending_graphs.add(new_graph)
                else:
                    if new_graph not in checked_graphs:
                        pending_graphs.add(new_graph)

        return False

    def num_subgraphs(self, exclude_qubits: set = set()) -> int:
        exclude_qubits = set(exclude_qubits)
        
        filter_qubits = lambda qubit: qubit not in exclude_qubits

        reduced_subgraph = nx.subgraph_view(self.network, filter_node=filter_qubits)

        return nx.number_connected_components(reduced_subgraph)

    def num_edges(self, exclude_qubits: set = set()) -> int:
        exclude_qubits = set(exclude_qubits)
        
        filter_qubits = lambda qubit: qubit not in exclude_qubits

        reduced_subgraph = nx.subgraph_view(self.network, filter_node=filter_qubits)

        return reduced_subgraph.number_of_edges()
    
    def type2_fusion(self, a: int, b: int, abar: int):
        if a not in self.qubit_list:
            raise Exception(f"{a} not in the graph")
        if b not in self.qubit_list:
            raise Exception(f"{b} not in the graph")
        if abar not in self.qubit_list:
            raise Exception(f"{abar} not in the graph")
        
        if not self.is_neighbor(abar, a):
            raise Exception(f"{a} and {abar} not neighbors")
        if len(self.get_nbhd(a).intersection(self.get_nbhd(b))) > 0:
            raise Exception(f"Our derivation assumes disjoint neighborhoods")
        
        qubits = set(self.qubit_list)
        N_a = self.get_nbhd(a)
        N_b = self.get_nbhd(b)
        N_abar = self.get_nbhd(abar)

        neighborhoods = [None]*self.n_qubits
        
        for i in qubits.difference(N_a.union(N_b).union(N_abar)):
            neighborhoods[i] = self.get_nbhd(i)

        for i in N_abar.difference(N_a.union(N_b)):
            neighborhoods[i] = self.get_nbhd(i).symmetric_difference(N_a.union(N_b))

        neighborhoods[abar] = (N_a.union(N_b)).difference({abar})

        for j in N_a.difference(N_abar.union({abar})):
            neighborhoods[j] = {abar}.union(self.get_nbhd(j).symmetric_difference(N_abar))

        for j in N_a.intersection(N_abar):
            neighborhoods[j] = {abar}.union(self.get_nbhd(j).symmetric_difference(N_abar).symmetric_difference(N_a.union(N_b)))

        for k in N_b.difference(N_abar):
            neighborhoods[k] = {abar}.union(self.get_nbhd(k).symmetric_difference(N_abar)).difference({a, b})

        for k in N_b.intersection(N_abar):
            neighborhoods[k] = {abar}.union(self.get_nbhd(k).symmetric_difference(N_abar).symmetric_difference(N_a.union(N_b))).difference({a, b})

        neighborhoods[a] = {}
        
        neighborhoods[b] = {}

        edges = set()
        for qubit in qubits:
            if qubit in neighborhoods[qubit]:
                print(neighborhoods[qubit])
                raise Exception(f"Qubit {qubit} in its own neighborhood!")

            for neighbor in neighborhoods[qubit]:
                if qubit not in neighborhoods[neighbor]:
                    raise Exception(f"Inconsistency in the relationship between {qubit} and {neighbor}!")

                if qubit < neighbor:
                    edges.add((qubit, neighbor))

        self.edges = edges
    
    # def measure_Z(self, qubit) -> None:
    #     for neighbor in self.get_nbhd(qubit):
    #         self.remove_edge(qubit, neighbor)
