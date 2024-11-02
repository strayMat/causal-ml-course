# %%
import networkx as nx
from pgmpy.base.DAG import DAG
import matplotlib.pyplot as plt
import pandas as pd
# %% [markdown]
# # Graph Generation and Plotting

# The following DAG is due to Judea Pearl.
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
# %%
nx.draw_planar(G, with_labels=True)
plt.show()
# %%
print(list(G.predecessors("X2")))
print(list(G.successors("X2")))
print(list(nx.ancestors(G, "X2")))
print(list(nx.descendants(G, "X2")))
# %%
# Find Paths Between D and Y
list(nx.all_simple_paths(G.to_undirected(), "D", "Y"))

# %%
# these returns all conditional independencies even among two sets of variables
# conditional on a third set
dseps = G.get_independencies()
# we display only the ones that correpond to pairs of singletons
for dsep in dseps.get_assertions():
    if len(dsep.get_assertion()[1]) == 1:
        print(dsep)

# %% [markdown]

# # Drawing a DAG for the wage dataset

# %%
file = (
    "https://raw.githubusercontent.com/CausalAIBook/"
    "MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
)
df = pd.read_csv(file)
# %%
import wooldridge

wooldridge.data("wageprc", description=True)
