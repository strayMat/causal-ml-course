# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Python: Real-Data Example for Multi-Period Difference-in-Differences
#
# In this example, we replicate a [real-data demo notebook](https://bcallaway11.github.io/did/articles/did-basics.html#an-example-with-real-data) from the [did-R-package](https://bcallaway11.github.io/did/index.html) in order to illustrate the use of `DoubleML` for multi-period difference-in-differences (DiD) models. 
#
#
#
# The notebook requires the following packages:

# %%
import pyreadr
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDMulti

# %% [markdown]
# ## Causal Research Question
#
# [Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) study the causal effect of raising the minimum wage on teen employment in the US using county data over a period from 2001 to 2007. A county is defined as treated if the minimum wage in that county is above the federal minimum wage. We focus on a preprocessed balanced panel data set as provided by the [did-R-package](https://bcallaway11.github.io/did/index.html). The corresponding documentation for the `mpdta` data is available from the [did package website](https://bcallaway11.github.io/did/reference/mpdta.html). We use this data solely as a demonstration example to help readers understand differences in the `DoubleML` and `did` packages. An analogous notebook using the same data is available from the [did documentation](https://bcallaway11.github.io/did/articles/did-basics.html#an-example-with-real-data).
#
# We follow the original notebook and provide results under identification based on unconditional and conditional parallel trends. For the Double Machine Learning (DML) Difference-in-Differences estimator, we demonstrate two different specifications, one based on linear and logistic regression and one based on their $\ell_1$ penalized variants Lasso and logistic regression with cross-validated penalty choice. The results for the former are expected to be very similar to those in the [did data example](https://bcallaway11.github.io/did/articles/did-basics.html#an-example-with-real-data). Minor differences might arise due to the use of sample-splitting in the DML estimation.
#
#
# ## Data
#
# We will download and read a preprocessed data file as provided by the [did-R-package](https://bcallaway11.github.io/did/index.html).
#

# %%
# download file from did package for R
url = "https://github.com/bcallaway11/did/raw/refs/heads/master/data/mpdta.rda"
pyreadr.download_file(url, "mpdta.rda")

mpdta = pyreadr.read_r("mpdta.rda")["mpdta"]
mpdta.head()


# %% [markdown]
# To work with [DoubleML](https://docs.doubleml.org/stable/index.html), we initialize a `DoubleMLPanelData` object. The input data has to satisfy some requirements, i.e., it should be in a *long* format with every row containing the information of one unit at one time period. Moreover, the data should contain a column on the unit identifier and a column on the time period. The requirements are virtually identical to those of the [did-R-package](https://bcallaway11.github.io/did/index.html), as listed in [their data example](https://bcallaway11.github.io/did/articles/did-basics.html#an-example-with-real-data). In line with the naming conventions of  [DoubleML](https://docs.doubleml.org/stable/index.html), the treatment group indicator is passed to `DoubleMLPanelData`  by the `d_cols` argument. To flexibly handle different formats for handling time periods, the time variable `t_col` can handle `float`, `int` and `datetime` formats. More information are available in the [user guide](https://docs.doubleml.org/dev/guide/data_backend.html#doublemlpaneldata). To indicate never treated units, we set their value for the treatment group variable to `np.inf`.

# %% [markdown]
# Now, we can initialize the ``DoubleMLPanelData`` object, specifying
#
#  - `y_col` : the outcome
#  - `d_cols`: the group variable indicating the first treated period for each unit
#  - `id_col`: the unique identification column for each unit
#  - `t_col` : the time column
#  - `x_cols`: the additional pre-treatment controls
#

# %%
# Set values for treatment group indicator for never-treated to np.inf
mpdta.loc[mpdta['first.treat'] == 0, 'first.treat'] = np.inf

dml_data = DoubleMLPanelData(
    data=mpdta,
    y_col="lemp",
    d_cols="first.treat",
    id_col="countyreal",
    t_col="year",
    x_cols=['lpop']
)
print(dml_data)

# %% [markdown]
# Note that we specified a pre-treatment confounding variable `lpop` through the `x_cols` argument. To consider cases under unconditional parallel trends, we can use dummy learners to ignore the pre-treatment confounding variable. This is illustrated below.

# %% [markdown]
# ## ATT Estimation: Unconditional Parallel Trends
#
# We start with identification under the unconditional parallel trends assumption. To do so, initialize a `DoubleMLDIDMulti` object (see [model documentation](https://docs.doubleml.org/stable/guide/models.html#difference-in-differences-models-did)), which takes the previously initialized `DoubleMLPanelData` object as input. We use scikit-learn's `DummyRegressor` (documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)) and `DummyClassifier` (documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)) to ignore the pre-treatment confounding variable. At this stage, we can also pass further options, for example specifying the number of folds and repetitions used for cross-fitting. 
#
# When calling the `fit()` method, the model estimates standard combinations of $ATT(g,t)$ parameters, which corresponds to the defaults in the [did-R-package](https://bcallaway11.github.io/did/index.html). These combinations can also be customized through the `gt_combinations` argument, see [the user guide](https://docs.doubleml.org/stable/guide/models.html#panel-data).

# %%
dml_obj = DoubleMLDIDMulti(
    obj_dml_data=dml_data,
    ml_g=DummyRegressor(),
    ml_m=DummyClassifier(),
    control_group="never_treated",
    n_folds=10
)

dml_obj.fit()
print(dml_obj.summary.round(4))

# %% [markdown]
# The summary displays estimates of the $ATT(g,t_\text{eval})$ effects for different combinations of $(g,t_\text{eval})$ via $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$, where
#  - $\mathrm{g}$ specifies the group
#  - $t_\text{pre}$ specifies the corresponding pre-treatment period
#  - $t_\text{eval}$ specifies the evaluation period
#
# This corresponds to the estimates given in `att_gt` function in the [did-R-package](https://bcallaway11.github.io/did/index.html), where the standard choice is $t_\text{pre} = \min(\mathrm{g}, t_\text{eval}) - 1$ (without anticipation).
#
# Remark that this includes pre-tests effects if $\mathrm{g} > t_{eval}$, e.g. $ATT(2007,2005)$.

# %% [markdown]
# As usual for the DoubleML-package, you can obtain joint confidence intervals via bootstrap.

# %%
level = 0.95

ci = dml_obj.confint(level=level)
dml_obj.bootstrap(n_rep_boot=5000)
ci_joint = dml_obj.confint(level=level, joint=True)
print(ci_joint)

# %% [markdown]
# A visualization of the effects can be obtained via the `plot_effects()` method.
#
# Remark that the plot used joint confidence intervals per default. 

# %% tags=["nbsphinx-thumbnail"]
fig, ax = dml_obj.plot_effects()

# %% [markdown]
# ## Effect Aggregation
#
# As the [did-R-package](https://bcallaway11.github.io/did/index.html), the $ATT$'s can be aggregated to summarize multiple effects.
# For details on different aggregations and details on their interpretations see [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001).
#
# The aggregations are implemented via the `aggregate()` method. We follow the structure of the [did package notebook](https://bcallaway11.github.io/did/articles/did-basics.html#an-example-with-real-data) and start with an aggregation relative to the treatment timing.

# %% [markdown]
# ### Event Study Aggregation
#
#
# We can aggregate the $ATT$s relative to the treatment timing. This is done by setting `aggregation="eventstudy"` in the `aggregate()` method. 
#  `aggregation="eventstudy"` aggregates $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$ based on exposure time $e = t_\text{eval} - \mathrm{g}$ (respecting group size).

# %%
# rerun bootstrap for valid simultaneous inference (as values are not saved) 
dml_obj.bootstrap(n_rep_boot=5000)
aggregated_eventstudy = dml_obj.aggregate("eventstudy")
# run bootstrap to obtain simultaneous confidence intervals
aggregated_eventstudy.aggregated_frameworks.bootstrap()
print(aggregated_eventstudy)
fig, ax = aggregated_eventstudy.plot_effects()

# %% [markdown]
# Alternatively, the $ATT$ could also be aggregated according to (calendar) time periods or treatment groups, see the [user guide](https://docs.doubleml.org/dev/guide/models.html#effect-aggregation).

# %% [markdown]
# ### Aggregation Details
#
# The `DoubleMLDIDAggregation` objects include several `DoubleMLFrameworks` which support methods like `bootstrap()` or `confint()`.
# Further, the weights can be accessed via the properties
#
#  - ``overall_aggregation_weights``: weights for the overall aggregation
#  - ``aggregation_weights``: weights for the aggregation
#
# To clarify, e.g. for the eventstudy aggregation

# %% [markdown]
# If one would like to consider how the aggregated effect with $e=0$ is computed, one would have to look at the third set of weights within the ``aggregation_weights`` property

# %%
aggregated_eventstudy.aggregation_weights[2]

# %% [markdown]
# ## ATT Estimation: Conditional Parallel Trends
#
# We briefly demonstrate how to use the `DoubleMLDIDMulti` model with conditional parallel trends. As the rationale behind DML is to flexibly model nuisance components as prediction problems, the DML DiD estimator includes pre-treatment covariates by default. In DiD, the nuisance components are the outcome regression and the propensity score estimation for the treatment group variable. This is why we had to enforce dummy learners in the unconditional parallel trends case to ignore the pre-treatment covariates. Now, we can replicate the classical doubly robust DiD estimator as of [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) by using linear and logistic regression for the nuisance components. This is done by setting `ml_g` to `LinearRegression()` and `ml_m` to `LogisticRegression()`. Similarly, we can also choose other learners, for example by setting `ml_g` and `ml_m` to `LassoCV()` and `LogisticRegressionCV()`. We present the results for the ATTs and their event-study aggregation in the corresponding effect plots.
#
# Please note that the example is meant to illustrate the usage of the `DoubleMLDIDMulti` model in combination with ML learners. In real-data applicatoins, careful choice and empirical evaluation of the learners are required. Default measures for the prediction of the nuisance components are printed in the model summary, as briefly illustrated below.

# %%
dml_obj_linear_logistic = DoubleMLDIDMulti(
    obj_dml_data=dml_data,
    ml_g=LinearRegression(),
    ml_m=LogisticRegression(penalty=None),
    control_group="never_treated",
    n_folds=10
)

dml_obj_linear_logistic.fit()
dml_obj_linear_logistic.bootstrap(n_rep_boot=5000)
dml_obj_linear_logistic.plot_effects(title="Estimated ATTs by Group, Linear and logistic Regression")


# %% [markdown]
# We briefly look at the model summary, which includes some standard diagnostics for the prediction of the nuisance components.

# %%
print(dml_obj_linear_logistic)

# %%
es_linear_logistic = dml_obj_linear_logistic.aggregate("eventstudy")
es_linear_logistic.aggregated_frameworks.bootstrap()
es_linear_logistic.plot_effects(title="Estimated ATTs by Group, Linear and logistic Regression")

# %%
dml_obj_lasso = DoubleMLDIDMulti(
    obj_dml_data=dml_data,
    ml_g=LassoCV(),
    ml_m=LogisticRegressionCV(),
    control_group="never_treated",
    n_folds=10
)

dml_obj_lasso.fit()
dml_obj_lasso.bootstrap(n_rep_boot=5000)
dml_obj_lasso.plot_effects(title="Estimated ATTs by Group, LassoCV and LogisticRegressionCV()")


# %%
# Model summary
print(dml_obj_lasso)

# %%
es_rf = dml_obj_lasso.aggregate("eventstudy")
es_rf.aggregated_frameworks.bootstrap()
es_rf.plot_effects(title="Estimated ATTs by Group, LassoCV and LogisticRegressionCV()")
