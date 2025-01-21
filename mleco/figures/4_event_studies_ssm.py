# %% 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# %%
# parallele trend assumptions
# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges for AR(1) process
G.add_edges_from([(t, t+1) for t in range(5)])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', arrowsize=20)
plt.title("Directed Acyclic Graph of AR(1)")
plt.show()
# %%