from instances import *
from environments import *
from solvers import *
from solutions import *

import networkx as nx
import matplotlib.pyplot as plt

def plot_solution_graph(Y_sol):
    n = Y_sol.shape[0]
    
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges for values of Y_sol == 1
    for i in range(n):
        for j in range(n):
            if Y_sol[i][j] == 1:
                G.add_edge(i, j)

    # Define layout (circular or spring)
    pos = nx.spring_layout(G, seed=42)  # or use nx.circular_layout(G)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{i}->{j}" for (i, j) in G.edges()})
    plt.title("Routing Solution")
    plt.show()


instance_name = 'dummy_problem'

inst = Instance(instance_name)
env = Environment(inst)
solver = solver_343420(env)

X, Y = solver.solve()

sol = Solution(X, Y)
sol.write(instance_name)
plot_solution_graph(Y)
