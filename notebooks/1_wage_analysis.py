# %%
import pandas as pd

# %% [markdown]
## Data Analysis

# We consider the same subsample of the U.S. Current Population Survey (2015). Let us load the data set.
# %%
file = (
    "https://raw.githubusercontent.com/CausalAIBook/"
    "MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
)
df = pd.read_csv(file)
# %%
df.head()
