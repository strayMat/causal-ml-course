# %%
import networkx as nx
from pgmpy.base.DAG import DAG
import matplotlib.pyplot as plt

# %%
digraph = nx.DiGraph(
    [
        ("Z1", "X1"),
        ("X1", "D"),
        ("Z1", "X2"),
        ("Z2", "X3"),
        ("X3", "Y"),
        ("Z2", "X2"),
        ("X2", "Y"),
        ("X2", "D"),
        ("M", "Y"),
        ("D", "M"),
    ]
)
# %%
G = DAG(digraph)
nx.draw_planar(G, with_labels=True)
plt.show()
# %%
print(list(G.predecessors("X2")))
print(list(G.successors("X2")))
print(list(nx.ancestors(G, "X2")))
print(list(nx.descendants(G, "X2")))
# %% [markdown]
# # Find paths between D and Y
# %%
list(nx.all_simple_paths(G.to_undirected(), "D", "Y"))
