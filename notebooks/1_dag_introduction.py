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
# These returns all conditional independencies even among two sets of variables
# conditional on a third set
dseps = G.get_independencies()
# we display only the ones that correpond to pairs of singletons
for dsep in dseps.get_assertions():
    if len(dsep.get_assertion()[1]) == 1:
        print(dsep)
# %% [markdown]
# # Simulation studies

# ## A fork of icecream under the sun

# [Credits to Prof. Reza Arghandeh](https://github.com/Ci2Lab/Applied_Causal_Inference_Course/blob/main/lectures/CH-3-Graphical-Causal-Models.ipynb)

# In hot summer months, people tend to consume more ice cream and are also more likely to get sunburns. While it might seem that Ice Cream Consumption (I) and Number of Sunburns (S) are related, this relationship is actually driven by a third variable, Hot Temperature (H).

# ### Causal graph for icecream and sunburns
#
# - TODO: Draw the graph the graph corresponding to the causal relationships between these variables.
# %%
digraph = nx.DiGraph(
    [
        ("Hot", "Ice Cream"),
        ("Hot", "Sunburst"),
    ]
)
G = DAG(digraph)
nx.draw_planar(G, with_labels=True)
plt.show()
# %% [markdown]
# ## Simulate the data for icecream and sunburns
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
# We now use a scatter plot matrix to visualize the pairwise relationships between Hot Temperature, Ice Cream Consumption, and Number of Sunburns.
# %%
# Visualizing the scatter plot matrix
sns.pairplot(data)
plt.suptitle(
    "Scatter Plot Matrix: Ice Cream Consumption, Sunburns, and Hot Temperature", y=1.02
)
plt.show()

# %% [markdown]
# ### Regression analysis for icecream and sunburns
# First let's begin by a naive regression analysis between Ice Cream Consumption and Number of Sunburns.

# %%
import statsmodels.formula.api as smf

# Resizualized with Hot Temperature
naive_fit = smf.ols("number_sunburns ~ ice_cream_consumption", data).fit()
print(naive_fit.summary())
# %% [markdown]
# We see that the coefficient for ice_cream_consumption is positive and statistically significant. This might lead us to believe that ice cream consumption causes sunburns. However, this is not the case. The relationship between ice cream consumption and sunburns is confounded by hot temperature. Let's see what happens when we control for hot temperature.
# %%
# Resizualized with Hot Temperature
# the ice_cream_consumption
ice_cream_ols = smf.ols("ice_cream_consumption ~ hot_temperature", data).fit()
print(ice_cream_ols.summary())
ice_cream_debiased = data["ice_cream_consumption"] - ice_cream_ols.predict(
    data[["hot_temperature"]]
)
# the sun_burns
sun_burns_ols = smf.ols("number_sunburns ~ hot_temperature", data).fit()
print(sun_burns_ols.summary())
sunburns_debiased = data["number_sunburns"] - sun_burns_ols.predict(
    data[["hot_temperature"]]
)
# %% [markdown]
# Lets's vizualize the debiased data.
# %%
residuals_data = pd.DataFrame(
    {
        "ice_cream_residuals": ice_cream_debiased,
        "sunburns_residuals": sunburns_debiased,
    }
)
# Visualize residuals (relationship after conditioning)
sns.pairplot(residuals_data)
plt.suptitle("Residuals (Conditioned on Temperature)", y=1.02)
plt.show()
# %% [markdown]
# Debiased regression
# %%
print(
    smf.ols("sunburns_residuals ~ ice_cream_residuals", residuals_data).fit().summary()
)
# %% [markdown]
# The coefficient for ice_cream_residuals is now close to zero and not statistically significant. This suggests that the relationship between ice cream consumption and number of sunburns is spurious and driven by hot temperature.
# %% [markdown]
# ## Collider bias for celebrities at Hollywood

# Here is a simple example to illustate the collider or M-bias.
# Credits to [Chernozhulov et al., Causal ML book](https://www.causalml-book.org/).

# The idea is that people who get to Hollywood have to have high congenility = talent + beauty. Funnily enough this induces a negative correlation between talents and looks, when we condition on the set of actors or celebrities.
#
# This simple example explains an anecdotal observation that "talent and beaty are negatively correlated" for celebrities.
# ### Causal graph for celebrities at Hollywood

# - TODO: Draw the graph the graph corresponding to the causal relationships between these variables.
# %%
digraph = nx.DiGraph([("T", "C"), ("B", "C")])
g = DAG(digraph)

nx.draw_planar(g, with_labels=True)
plt.show()
# %% [markdown]
# ### Simulate the data for celebrities at Hollywood
# %%
np.random.seed(123)
num_samples = 1000000
talent = np.random.normal(size=num_samples)
beauty = np.random.normal(size=num_samples)
congeniality = talent + beauty + np.random.normal(size=num_samples)  # congeniality
cond_talent = talent[congeniality > 0]
cond_beauty = beauty[congeniality > 0]
data = {
    "talent": talent,
    "beauty": beauty,
    "congeniality": congeniality,
    "cond_talent": cond_talent,
    "cond_beauty": cond_beauty,
}
# %% [markdown]
# ### Visualize the relationships for celebrities at Hollywood
# - TODO: Visualize the scatter plot matrix for the data.
# %%

# %% [markdown]
# ### Regression analysis

# - TODO: Perform regression analysis to show the collider bias.
print(smf.ols("talent ~ beauty", data).fit().summary())
print(smf.ols("talent ~ beauty + congeniality", data).fit().summary())
print(smf.ols("cond_talent ~ cond_beauty", data).fit().summary())
