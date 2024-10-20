# %%
import networkx as nx

digraph = nx.DiGraph([('Z1', 'X1'),
                      ('X1', 'D'),
                      ('Z1', 'X2'),
                      ('Z2', 'X3'),
                      ('X3', 'Y'),
                      ('Z2', 'X2'),
                      ('X2', 'Y'),
                      ('X2', 'D'),
                      ('M', 'Y'),
                      ('D', 'M')])
# %%
from pgmpy.base.DAG import DAG

G = DAG(digraph)