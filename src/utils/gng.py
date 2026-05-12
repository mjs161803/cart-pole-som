import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.node import Node
from utils.edge import Edge   

class GNG:
    def __init__(self, data, epsilon_b=0.05, epsilon_n=0.001, lambda_=100, alpha=0.5, max_nodes=100):
        self.nodes = {}  # node index -> Node
        self.edges = []
        self.edge_lookup = {}  # frozenset({i, j}) -> Edge
        self.adjacency = {}  # node index -> set of neighbor indices
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.lambda_ = lambda_
        self.alpha = alpha # Decrease factor for accumulated error when inserting new nodes
        self.global_decay = 0.995 # Global decay factor for accumulated error after each iteration
        self.max_nodes = max_nodes
        self._next_index = 0  # monotonically increasing node index counter

        # 1) Initialize the GNG with two random nodes from the input data

           # 1.1 Find max/min values for each element in the input data
        self.max_data_values = np.max(data, axis=0)
        self.min_data_values = np.min(data, axis=0)

           # 1.2 Initialize a random weights bounded by min/max values per dimension
        initial_node1_weights = np.random.rand(data.shape[1]) * (self.max_data_values - self.min_data_values) + self.min_data_values
        initial_node2_weights = np.random.rand(data.shape[1]) * (self.max_data_values - self.min_data_values) + self.min_data_values
        n0 = Node(initial_node1_weights, self._next_index); self._next_index += 1
        n1 = Node(initial_node2_weights, self._next_index); self._next_index += 1
        self.nodes[n0.index] = n0
        self.nodes[n1.index] = n1
        initial_edge = Edge(n0.index, n1.index)
        self.edges.append(initial_edge)
        self.edge_lookup[frozenset({n0.index, n1.index})] = initial_edge
        self.adjacency[n0.index] = {n1.index}
        self.adjacency[n1.index] = {n0.index}

    def fit(self, data, max_iterations=1000):
        for iteration in range(max_iterations):
            # 0) Guard against degenerate state
            if len(self.nodes) < 2:
                break

            # 1) Randomly select an input signal from the data
            sample = data[np.random.randint(0, data.shape[0])]

            # 2) Find the two nearest nodes (s1 and s2) to the input signal
            node_indices = list(self.nodes.keys())
            distances = {idx: np.linalg.norm(self.nodes[idx].weights - sample) for idx in node_indices}
            sorted_indices = sorted(distances, key=distances.__getitem__)
            s1_index, s2_index = sorted_indices[0], sorted_indices[1]
            s1 = self.nodes[s1_index]

            # 3) Increment the age of all edges emanating from s1
            for neighbor_index in self.adjacency[s1_index]:
                edge = self.get_edge(s1_index, neighbor_index)
                if edge:
                    edge.age += 1

            # 4) Update the accumulated error of s1
            s1.accumulated_error += distances[s1_index] ** 2

            # 5) Move s1 and its topological neighbors towards the input signal
            s1.weights += self.epsilon_b * (sample - s1.weights)
            for neighbor_index in self.adjacency[s1_index]:
                self.nodes[neighbor_index].weights += self.epsilon_n * (sample - self.nodes[neighbor_index].weights)

            # 6) If s1 and s2 are connected by an edge, reset the age of that edge to 0. Otherwise, create a new edge between s1 and s2
            edge = self.get_edge(s1_index, s2_index)
            if edge:
                edge.age = 0
            else:
                new_edge = Edge(s1_index, s2_index)
                self.edges.append(new_edge)
                self.edge_lookup[frozenset({s1_index, s2_index})] = new_edge
                self.adjacency[s1_index].add(s2_index)
                self.adjacency[s2_index].add(s1_index)

            # 7) Remove edges that have an age greater than a predefined maximum age. If this results in nodes having no emanating edges, remove those nodes as well
            for edge in self.edges:
                if edge.age > edge.max_age:
                    self.edge_lookup.pop(frozenset({edge.node1_index, edge.node2_index}), None)
            self.edges = [edge for edge in self.edges if edge.age <= edge.max_age]
            for node_index in list(self.adjacency.keys()):
                self.adjacency[node_index] = {nb for nb in self.adjacency[node_index] if self.get_edge(node_index, nb) is not None}
                if not self.adjacency[node_index]:  # If no neighbors left, remove the node
                    del self.adjacency[node_index]
                    del self.nodes[node_index]

            # 8) Every λ iterations, insert a new node
            if (iteration + 1) % self.lambda_ == 0 and len(self.nodes) < self.max_nodes:
                # 8.1) Find the node q with the largest accumulated error
                q_index = max(self.nodes, key=lambda idx: self.nodes[idx].accumulated_error)
                q = self.nodes[q_index]
                # 8.2) Among the neighbors of q, find the node f with the largest accumulated error
                if self.adjacency.get(q_index):  # Ensure q has neighbors
                    f_index = max(self.adjacency[q_index], key=lambda idx: self.nodes[idx].accumulated_error)
                    f = self.nodes[f_index]
                    # 8.3) Insert a new node r halfway between q and f
                    r_weights = (q.weights + f.weights) / 2
                    r_index = self._next_index; self._next_index += 1
                    r = Node(r_weights, r_index)
                    self.nodes[r_index] = r
                    # 8.4) Remove the edge between q and f, and create edges between r and q, and between r and f
                    self.edge_lookup.pop(frozenset({q_index, f_index}), None)
                    self.edges = [e for e in self.edges if not ((e.node1_index == q_index and e.node2_index == f_index) or (e.node1_index == f_index and e.node2_index == q_index))]
                    self.adjacency[q_index].discard(f_index)
                    self.adjacency[f_index].discard(q_index)
                    qr_edge = Edge(q_index, r_index)
                    fr_edge = Edge(f_index, r_index)
                    self.edges.append(qr_edge)
                    self.edges.append(fr_edge)
                    self.edge_lookup[frozenset({q_index, r_index})] = qr_edge
                    self.edge_lookup[frozenset({f_index, r_index})] = fr_edge
                    self.adjacency.setdefault(q_index, set()).add(r_index)
                    self.adjacency.setdefault(f_index, set()).add(r_index)
                    self.adjacency[r_index] = {q_index, f_index}
                    # 8.5) Decrease the accumulated error of q and f by multiplying them with a constant α, and initialize the accumulated error of r to be equal to the new accumulated error of q
                    q.accumulated_error *= self.alpha
                    f.accumulated_error *= self.alpha
                    r.accumulated_error = q.accumulated_error

            # 9) Decrease the accumulated error of all nodes by multiplying them with a global decay factor, and increment accumulated_age
            for node in self.nodes.values():
                node.accumulated_error *= self.global_decay
                node.accumulated_age += 1

            # 10) Stop the algorithm if a predefined maximum number of nodes is reached
            if len(self.nodes) >= self.max_nodes:
                break

    def visualize(self, title="GNG Structure"):
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.index, error=node.accumulated_error)
        for edge in self.edges:
            G.add_edge(edge.node1_index, edge.node2_index, age=edge.age)

        pos = nx.spring_layout(G, seed=42)
        errors = [G.nodes[n]['error'] for n in G.nodes]
        ages = [G.edges[e]['age'] for e in G.edges]

        fig, ax = plt.subplots(figsize=(10, 7))
        if ages:
            nx.draw_networkx_edges(G, pos, edge_color=ages, edge_cmap=plt.cm.cool, ax=ax)
        nodes_drawn = nx.draw_networkx_nodes(G, pos, node_color=errors, cmap=plt.cm.YlOrRd, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        if nodes_drawn is not None:
            plt.colorbar(nodes_drawn, ax=ax, label="Accumulated Error")
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def get_edge(self, node1_index, node2_index):
        return self.edge_lookup.get(frozenset({node1_index, node2_index}))
    
        
    
