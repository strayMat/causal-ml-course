# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Supplementary practical session focused on synthetic control method.
# 
#  Application of the synthetic control method to the German reunification: using the `pysyncon` package, [example notebook from the package](https://github.com/sdfordham/pysyncon/blob/main/examples/germany.ipynb). 
# 
# This notebook reproduces the weights from the 2015 paper "Comparative Politics and the Synthetic Control Method" by Alberto Abadie, Alexis Diamond and Jens Hainmueller. The study data is contained in the file `../data/germany.csv` (more detailed information on this dataset is available in the appendix of the cited paper). This notebook is taken from the `pysyncon` [package](https://github.com/sdfordham/pysyncon).

# %%
import pandas as pd
from pysyncon import Dataprep, Synth

# %% [markdown]
# The study is carried out in two steps. In each step we prepare the study data using a `Dataprep` object that describes all the details needed to define the synthetic control study. This works similarly to the `dataprep` method in the `R` package `synth`.
#
# In the first run, the optimisation is carried out over the period 1981 to 1991, and the $V$ matrix obtained in this optimisation is then used in the second run, where the optimisation is carried out over the period 1960 to 1990. This serves to enforce that the predictor importances used in the final optimisation are those of the eighties.
#
# In each case, we supply the `Dataprep` object to a `Synth` object. In the second run, we can provide a custom $V$ matrix with the `custom_V` option.
#
# (For an explanation of each of the `Dataprep` arguments, see the package [documentation](https://sdfordham.github.io/pysyncon/dataprep.html#pysyncon.Dataprep)).

# %%
df = pd.read_csv("../data/germany.csv")

dataprep_train = Dataprep(
    foo=df,
    predictors=["gdp", "trade", "infrate"],
    predictors_op="mean",
    time_predictors_prior=range(1971, 1981),
    special_predictors=[
        ("industry", range(1971, 1981), "mean"),
        ("schooling", [1970, 1975], "mean"),
        ("invest70", [1980], "mean"),
    ],
    dependent="gdp",
    unit_variable="country",
    time_variable="year",
    treatment_identifier="West Germany",
    controls_identifier=[
        "USA",
        "UK",
        "Austria",
        "Belgium",
        "Denmark",
        "France",
        "Italy",
        "Netherlands",
        "Norway",
        "Switzerland",
        "Japan",
        "Greece",
        "Portugal",
        "Spain",
        "Australia",
        "New Zealand",
    ],
    time_optimize_ssr=range(1981, 1991),
)

print(dataprep_train)

# %%
synth_train = Synth()
synth_train.fit(dataprep=dataprep_train)

# %%
dataprep = Dataprep(
    foo=df,
    predictors=["gdp", "trade", "infrate"],
    predictors_op="mean",
    time_predictors_prior=range(1981, 1991),
    special_predictors=[
        ("industry", range(1981, 1991), "mean"),
        ("schooling", [1980, 1985], "mean"),
        ("invest80", [1980], "mean"),
    ],
    dependent="gdp",
    unit_variable="country",
    time_variable="year",
    treatment_identifier="West Germany",
    controls_identifier=[
        "USA",
        "UK",
        "Austria",
        "Belgium",
        "Denmark",
        "France",
        "Italy",
        "Netherlands",
        "Norway",
        "Switzerland",
        "Japan",
        "Greece",
        "Portugal",
        "Spain",
        "Australia",
        "New Zealand",
    ],
    time_optimize_ssr=range(1960, 1990),
)

print(dataprep)

# %%
synth = Synth()
synth.fit(dataprep=dataprep, custom_V=synth_train.V)

synth.weights()

# %% [markdown]
# The synthetic control obtained from the optimisation is: $$\text{Synthetic Control} = 0.216 \times \text{USA} + 0.415 \times \text{Austria} + 0.098 \times \text{Nederlands} + 0.108 \times \text{Switzerland} + 0.163 \times \text{Japan}.$$

# %% [markdown]
# The `path_plot` method shows the path of the treated unit and the synthetic control over time.

# %%
synth.path_plot(time_period=range(1960, 2004), treatment_time=1990)

# %% [markdown]
# The `gaps_plot` method shows the gaps (the difference between the treated unit and the synthetic control) over time.

# %%
synth.gaps_plot(time_period=range(1960, 2004), treatment_time=1990)

# %% [markdown]
# The summary function give more information on the predictor values. The first column shows the value of the $V$ matrix for each predictor, the column 'treated' shows the mean value of each predictor for the treated unit over the time period `time_predictors_prior`, the column 'synthetic' shows the mean value of each predictor for the synthetic control over the time period `time_predictors_prior` and finally the column 'sample mean' shows the sample mean of that predictor for all control units over the time period `time_predictors_prior` i.e. this is the same as the synthetic control with all weights equal.

# %%
synth.summary()

# %% [markdown]
# Compute the average treatment effect on the treated unit (ATT) over the post-treatment time period. This method returns a standard error also.

# %%
synth.att(time_period=range(1990, 2004))

# %% [markdown]
# Calculate 95% confidence intervals for the treatment effect for time periods $t=1991, \ldots, 2000$.

# %%
synth.confidence_interval(
    alpha=0.05,
    time_periods=[1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
    custom_V=synth_train.V,
    tol=0.01,
    verbose=False,
)
