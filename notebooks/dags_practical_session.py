# %%
import networkx as nx
from pgmpy.base.DAG import DAG
import matplotlib.pyplot as plt
import pandas as pd
# %% [markdown]

# In this practical session, we will cover the following topics:
# - Graph Generation and Plotting
# - Simulation studies for common causal fallacies
#   - Fork paths
#   - Collider bias

# # Graph Generation and Plotting

# The following DAG is due to Judea Pearl.
# We will use it to illustrate how to generate a graph and plot it.
# We will also show how to compute the ancestors and descendants of a node.
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
# These returns all conditional independencies even among two sets of variables
# conditional on a third set
dseps = G.get_independencies()
# we display only the ones that correpond to pairs of singletons
for dsep in dseps.get_assertions():
    if len(dsep.get_assertion()[1]) == 1:
        print(dsep)
# %% [markdown]
# # Simulation studies
#
# ## A fork of icecream under the sun
#
# [Credits to Prof. Reza Arghandeh](https://github.com/Ci2Lab/Applied_Causal_Inference_Course/blob/main/lectures/CH-3-Graphical-Causal-Models.ipynb)
#
# In hot summer months, people tend to consume more ice cream and are also more
# likely to get sunburns. While it might seem that Ice Cream Consumption (I) and
#  Number of Sunburns (S) are related, this relationship is actually driven by a
# third variable, Hot Temperature (H).

# ### Causal graph for icecream and sunburns
#
# - TODO: Draw the causal graph corresponding to the causal relationships
# between these variables.
# %%
# CODE HERE
# %% [markdown]
# ### Simulate the data for icecream and sunburns
# Let's generate synthetic data for the three variables:
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Simulating Hot Temperature (T)
hot_temperature = np.random.normal(
    loc=30, scale=5, size=100
)  # Average temperature of 30 degrees with some variation

# Simulating Ice Cream Consumption (I) based on Temperature (T)
ice_cream_consumption = 2 * hot_temperature + np.random.normal(
    loc=0, scale=5, size=100
)  # Higher temperature, more ice cream

# Simulating Number of Sunburns (S) based on Temperature (T)
number_of_sunburns = 1.5 * hot_temperature + np.random.normal(
    loc=0, scale=3, size=100
)  # Higher temperature, more sunburns

# Creating a DataFrame for the data
data = pd.DataFrame(
    {
        "hot_temperature": hot_temperature,
        "ice_cream_consumption": ice_cream_consumption,
        "number_sunburns": number_of_sunburns,
    }
)

# Display the first few rows of the dataset
data.head()
# %% [markdown]
# ### Visualize the relationships between icecream and sunburns
# We now use a scatter plot matrix to visualize the pairwise relationships
# between Hot Temperature, Ice Cream Consumption, and Number of Sunburns.
# - TODO: Visualize the scatter plot matrix for the data. You can use the
# pairplot function from seaborn.
# %%
# CODE HERE
# %% [markdown]
# ### Regression analysis for icecream and sunburns
# First let's begin by a naive regression analysis between Ice Cream Consumption
#  and Number of Sunburns.
# %%
import statsmodels.formula.api as smf

# Resizualized with Hot Temperature
naive_fit = smf.ols("number_sunburns ~ ice_cream_consumption", data).fit()
print(naive_fit.summary())
# %% [markdown]
# We see that the coefficient for ice_cream_consumption is positive and
# statistically significant. This might lead us to believe that ice cream
# consumption causes sunburns. However, this is not the case. The relationship
#  between ice cream consumption and sunburns is confounded by hot temperature.
#  Let's see what happens when we control for hot temperature.

# ### Resizualized with Hot Temperature
# - TODO: Perform a regression analysis for both the ice cream and the number of
#  sun burns to *regress away* the effect of the hot_temperature.
# Regression for ice_cream_consumption
# %%
# CODE HERE
# %% [markdown]
# Lets's vizualize the debiased data.
# %%
residuals_data = pd.DataFrame(
    {
        "ice_cream_residuals": ice_cream_debiased,
        "sunburns_residuals": sunburns_debiased,
    }
)
# %% [markdown]
# ## Visualize the residuals (relationship after conditioning)
# %%
# CODE HERE
# %% [markdown]
# ## Debiased regression
# %%
print(
    smf.ols("sunburns_residuals ~ ice_cream_residuals", residuals_data).fit().summary()
)
# %% [markdown]
# TODO: What do you observe in the debiased regression analysis? WHat value do the coefficient of ice_cream_residuals take? What does it suggest for the relationship between ice cream consumption and number of sunburns?
# %% [markdown]
# # Collider bias for celebrities at Hollywood
#
# Here is a simple example to illustate the collider or M-bias.
# Credits to [Chernozhulov et al., 2024, Causal ML book](https://www.causalml-book.org/).
#
# The idea is that people who get to Hollywood tend to have a high
# congenility = talent + beauty. Funnily enough this induces a negative
# correlation between talents and looks, when we condition on the set of actors
# or celebrities.
#
# This simple example explains an anecdotal observation that "talent and beauty
# are negatively correlated" for celebrities.
# This is a form of collider bias, also coined as selection bias for this specific case.
#
# ### Causal graph for celebrities at Hollywood

# - TODO: Draw the graph the graph corresponding to the causal relationships between these variables.
# %%
# CODE HERE
# %% [markdown]
# ### Simulate the data for celebrities at Hollywood
# %%
np.random.seed(123)
num_samples = 10000
talent = np.random.normal(size=num_samples)
beauty = np.random.normal(size=num_samples)
congeniality = talent + beauty + np.random.normal(size=num_samples)  # congeniality
hollywood_data = pd.DataFrame(
    {
        "talent": talent,
        "beauty": beauty,
        "congeniality": congeniality,
    }
)
# Create the conditional variable: celebrity is True if congeniality > 2
hollywood_data["celebrity"] = hollywood_data["congeniality"] > 2
# %% [markdown]
# ### Visualize the relationships for celebrities at Hollywood
# - TODO: Visualize the pairplot :
#  - for the whole data.
#  - only for the celebrities.
# %%
# CODE HERE
# %% [markdown]
# We see that for the whole data, there is no correlation between talent and
# beauty. However, when we condition on the set of celebrities, we see a
# negative correlation between talent and beauty. This is an example of collider
# bias.

# ### Regression analysis for celebrities at Hollywood
# Recover what we have seen in the pairplot, that is, the negative correlation
# between talent and beauty for celebrities.
# - TODO: Perform regression analysis to show the collider bias. You should
# contrast a regression analysis for the whole data and for the celebrities only.
# %%
# CODE HERE