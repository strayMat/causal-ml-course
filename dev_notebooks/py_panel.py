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
# # Python: Panel Data with Multiple Time Periods
#
# In this example, a detailed guide on Difference-in-Differences with multiple time periods using the [DoubleML-package](https://docs.doubleml.org/stable/index.html). The implementation is based on [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001).
#
# The notebook requires the following packages:

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.did import DoubleMLDIDMulti
from doubleml.data import DoubleMLPanelData

from doubleml.did.datasets import make_did_CS2021

# %% [markdown]
# ## Data
#
# We will rely on the `make_did_CS2021` DGP, which is inspired by [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) (Appendix SC) and [Sant'Anna and Zhao (2020)](https://doi.org/10.1016/j.jeconom.2020.06.003).
#
# We will observe `n_obs` units over `n_periods`. Remark that the dataframe includes observations of the potential outcomes `y0` and `y1`, such that we can use oracle estimates as comparisons.

# %%
n_obs = 5000
n_periods = 6

df = make_did_CS2021(n_obs, dgp_type=4, n_periods=n_periods, n_pre_treat_periods=3, time_type="datetime")
df["ite"] = df["y1"] - df["y0"]

print(df.shape)
df.head()

# %% [markdown]
# ### Data Details
#
# Here, we slightly abuse the definition of the potential outcomes. $Y_{i,t}(1)$ corresponds to the (potential) outcome if unit $i$ would have received treatment at time period $\mathrm{g}$ (where the group $\mathrm{g}$ is drawn with probabilities based on $Z$).
#
# More specifically
#
# $$
# \begin{align*}
# Y_{i,t}(0)&:= f_t(Z) + \delta_t + \eta_i + \varepsilon_{i,t,0}\\
# Y_{i,t}(1)&:= Y_{i,t}(0) + \theta_{i,t,\mathrm{g}} + \epsilon_{i,t,1} - \epsilon_{i,t,0}
# \end{align*}
# $$
#
# where
#  - $f_t(Z)$ depends on pre-treatment observable covariates $Z_1,\dots, Z_4$ and time $t$
#  - $\delta_t$ is a time fixed effect
#  - $\eta_i$ is a unit fixed effect
#  - $\epsilon_{i,t,\cdot}$ are time varying unobservables (iid. $N(0,1)$)
#  - $\theta_{i,t,\mathrm{g}}$ correponds to the exposure effect of unit $i$ based on group $\mathrm{g}$ at time $t$
#
# For the pre-treatment periods the exposure effect is set to
# $$
# \theta_{i,t,\mathrm{g}}:= 0 \text{ for } t<\mathrm{g}
# $$
# such that
#
# $$
# \mathbb{E}[Y_{i,t}(1) - Y_{i,t}(0)] = \mathbb{E}[\epsilon_{i,t,1} - \epsilon_{i,t,0}]=0  \text{ for } t<\mathrm{g}
# $$
#
# The [DoubleML Coverage Repository](https://docs.doubleml.org/doubleml-coverage/) includes coverage simulations based on this DGP.

# %% [markdown]
# ### Data Description

# %% [markdown]
# The data is a balanced panel where each unit is observed over `n_periods` starting Janary 2025.

# %%
df.groupby("t").size()

# %% [markdown]
# The treatment column `d` indicates first treatment period of the corresponding unit, whereas `NaT` units are never treated.
#
# *Generally, never treated units should take either on the value `np.inf` or `pd.NaT` depending on the data type (`float` or `datetime`).*
#
# The individual units are roughly uniformly divided between the groups, where treatment assignment depends on the pre-treatment covariates `Z1` to `Z4`.

# %%
df.groupby("d", dropna=False).size()

# %% [markdown]
# Here, the group indicates the first treated period and `NaT` units are never treated.
# %% [markdown]
# To get a better understanding of the underlying data and true effects, we will compare the unconditional averages and the true effects based on the oracle values of individual effects `ite`.

# %%
# rename for plotting
df["First Treated"] = df["d"].dt.strftime("%Y-%m").fillna("Never Treated")

# Create aggregation dictionary for means
def agg_dict(col_name):
    return {
        f'{col_name}_mean': (col_name, 'mean'),
        f'{col_name}_lower_quantile': (col_name, lambda x: x.quantile(0.05)),
        f'{col_name}_upper_quantile': (col_name, lambda x: x.quantile(0.95))
    }

# Calculate means and confidence intervals
agg_dictionary = agg_dict("y") | agg_dict("ite")

agg_df = df.groupby(["t", "First Treated"]).agg(**agg_dictionary).reset_index()
agg_df.head()


# %%
def plot_data(df, col_name='y', plot_full_data=False):
    """
    Create an improved plot with colorblind-friendly features

    Parameters:
    -----------
    df : DataFrame
        The dataframe containing the data
    col_name : str, default='y'
        Column name to plot (will use '{col_name}_mean')
    """
    plt.figure(figsize=(12, 7))
    n_colors = df["First Treated"].nunique()
    color_palette = sns.color_palette("colorblind", n_colors=n_colors)
    if plot_full_data:
        y_col = col_name
    else:
        y_col = f'{col_name}_mean'
    sns.lineplot(
        data=df,
        x='t',
        y=y_col,
        hue='First Treated',
        style='First Treated',
        palette=color_palette,
        markers=True,
        dashes=True,
        linewidth=2.5,
        alpha=0.8
    )

    plt.title(f'Average Values {col_name} by Group Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel(f'Average Value {col_name}', fontsize=14)


    plt.legend(title='First Treated', title_fontsize=13, fontsize=12,
               frameon=True, framealpha=0.9, loc='best')

    plt.grid(alpha=0.3, linestyle='-')
    plt.tight_layout()

    plt.show()

# %% [markdown]
# So let us take a look at the average values over time

# %%
plot_data(agg_df, col_name='y')
# %%
# plot full data
plot_data(df, col_name='y', plot_full_data=True)
# %% [markdown]
# Instead the true average treatment treatment effects can be obtained by averaging (usually unobserved) the `ite` values.
#
# The true effect just equals the exposure time (in months):
#
# $$
# ATT(\mathrm{g}, t) = \min(\mathrm{t} - \mathrm{g} + 1, 0) =: e
# $$
#

# %% jupyter={"source_hidden": true}
plot_data(agg_df, col_name='ite')

# %% [markdown]
# ### DoubleMLPanelData
#
# Finally, we can construct our `DoubleMLPanelData`, specifying
#
#  - `y_col` : the outcome
#  - `d_cols`: the group variable indicating the first treated period for each unit
#  - `id_col`: the unique identification column for each unit
#  - `t_col` : the time column
#  - `x_cols`: the additional pre-treatment controls
#  - `datetime_unit`: unit required for `datetime` columns and plotting

# %%
dml_data = DoubleMLPanelData(
    data=df,
    y_col="y",
    d_cols="d",
    id_col="id",
    t_col="t",
    x_cols=["Z1", "Z2", "Z3", "Z4"],
    datetime_unit="M"
)
print(dml_data)

# %% [markdown]
# ## ATT Estimation
#
# The [DoubleML-package](https://docs.doubleml.org/stable/index.html) implements estimation of group-time average treatment effect via the `DoubleMLDIDMulti` class (see [model documentation](https://docs.doubleml.org/stable/guide/models.html#difference-in-differences-models-did)).

# %% [markdown]
# ### Basics
#
# The class basically behaves like other `DoubleML` classes and requires the specification of two learners (for more details on the regression elements, see [score documentation](https://docs.doubleml.org/stable/guide/scores.html#difference-in-differences-models)).
#
# The basic arguments of a `DoubleMLDIDMulti` object include
#
#  - `ml_g` "outcome" regression learner
#  - `ml_m` propensity Score learner
#  - `control_group` the control group for the parallel trend assumption
#  - `gt_combinations` combinations of $(\mathrm{g},t_\text{pre}, t_\text{eval})$
#  - `anticipation_periods` number of anticipation periods
#
# We will construct a `dict` with "default" arguments.

# %%
default_args = {
    "ml_g": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.01, verbose=0, random_state=123),
    "ml_m": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.01, verbose=0, random_state=123),
    "control_group": "never_treated",
    "gt_combinations": "standard",
    "anticipation_periods": 0,
    "n_folds": 5,
    "n_rep": 1,
}

# %% [markdown]
#  The model will be estimated using the `fit()` method.

# %%
np.random.seed(42)
dml_obj = DoubleMLDIDMulti(dml_data, **default_args)
dml_obj.fit()
print(dml_obj)

# %% [markdown]
# The summary displays estimates of the $ATT(g,t_\text{eval})$ effects for different combinations of $(g,t_\text{eval})$ via $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$, where
#  - $\mathrm{g}$ specifies the group
#  - $t_\text{pre}$ specifies the corresponding pre-treatment period
#  - $t_\text{eval}$ specifies the evaluation period
#
# The choice `gt_combinations="standard"`, used estimates all possible combinations of $ATT(g,t_\text{eval})$ via $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$,
# where the standard choice is $t_\text{pre} = \min(\mathrm{g}, t_\text{eval}) - 1$ (without anticipation).
#
# Remark that this includes pre-tests effects if $\mathrm{g} > t_{eval}$, e.g. $\widehat{ATT}(g=\text{2025-04}, t_{\text{pre}}=\text{2025-01}, t_{\text{eval}}=\text{2025-02})$ which estimates the pre-trend from January to February even if the actual treatment occured in April.

# %% [markdown]
# As usual for the DoubleML-package, you can obtain joint confidence intervals via bootstrap.

# %%
level = 0.95

ci = dml_obj.confint(level=level)
dml_obj.bootstrap(n_rep_boot=5000)
ci_joint = dml_obj.confint(level=level, joint=True)
ci_joint

# %% [markdown]
# A visualization of the effects can be obtained via the `plot_effects()` method.
#
# Remark that the plot used joint confidence intervals per default.

# %% tags=["nbsphinx-thumbnail"]
dml_obj.plot_effects()

# %% [markdown]
# ### Sensitivity Analysis
#
# As descripted in the [Sensitivity Guide](https://docs.doubleml.org/stable/guide/sensitivity.html), robustness checks on omitted confounding/parallel trend violations are available, via the standard `sensitivity_analysis()` method.

# %%
dml_obj.sensitivity_analysis()
print(dml_obj.sensitivity_summary)

# %% [markdown]
# In this example one can clearly, distinguish the robustness of the non-zero effects vs. the pre-treatment periods.

# %% [markdown]
# ### Control Groups
#
# The current implementation support the following control groups
#
#  - ``"never_treated"``
#  - ``"not_yet_treated"``
#
# Remark that the ``"not_yet_treated" depends on anticipation.
#
# For differences and recommendations, we refer to [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001).

# %%
dml_obj_nyt = DoubleMLDIDMulti(dml_data, **(default_args | {"control_group": "not_yet_treated"}))
dml_obj_nyt.fit()
dml_obj_nyt.bootstrap(n_rep_boot=5000)
dml_obj_nyt.plot_effects()

# %% [markdown]
# ### Linear Covariate Adjustment
#
# Remark that we relied on boosted trees to adjust for conditional parallel trends which allow for a nonlinear adjustment. In comparison to linear adjustment, we could rely on linear learners.
#
# **Remark that the DGP (`dgp_type=4`) is based on nonlinear conditional expectations such that the estimates will be biased**
#
#

# %%
linear_learners = {
    "ml_g": LinearRegression(),
    "ml_m": LogisticRegression(),
}

dml_obj_linear = DoubleMLDIDMulti(dml_data, **(default_args | linear_learners))
dml_obj_linear.fit()
dml_obj_linear.bootstrap(n_rep_boot=5000)
dml_obj_linear.plot_effects()

# %% [markdown]
# ## Aggregated Effects
# As the [did-R-package](https://bcallaway11.github.io/did/index.html), the $ATT$'s can be aggregated to summarize multiple effects.
# For details on different aggregations and details on their interpretations see [Callaway and Sant'Anna(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001).
#
# The aggregations are implemented via the `aggregate()` method.

# %% [markdown]
# ### Group Aggregation
#
#
# To obtain group-specific effects one can would like to average $ATT(\mathrm{g}, t_\text{eval})$ over $t_\text{eval}$.
# As a sample oracle we will combine all `ite`'s based on group $\mathrm{g}$.

# %%
df_post_treatment = df[df["t"] >= df["d"]]
df_post_treatment.groupby("d")["ite"].mean()

# %% [markdown]
# To obtain group-specific effects it is possible to aggregate several $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$ values based on the group $\mathrm{g}$ by setting the `aggregation="group"` argument.

# %%
aggregated_group = dml_obj.aggregate(aggregation="group")
print(aggregated_group)
_ = aggregated_group.plot_effects()

# %% [markdown]
# The output is a `DoubleMLDIDAggregation` object which includes an overall aggregation summary based on group size.

# %% [markdown]
# ### Time Aggregation
#
# To obtain time-specific effects one can would like to average $ATT(\mathrm{g}, t_\text{eval})$ over $\mathrm{g}$ (respecting group size).
# As a sample oracle we will combine all `ite`'s based on group $\mathrm{g}$. As oracle values, we obtain

# %%
df_post_treatment.groupby("t")["ite"].mean()

# %% [markdown]
# To aggregate $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$, based on $t_\text{eval}$, but weighted with respect to group size. Corresponds to *Calendar Time Effects* from the [did-R-package](https://bcallaway11.github.io/did/index.html).
#
# For calendar time effects set `aggregation="time"`.

# %%
aggregated_time = dml_obj.aggregate("time")
print(aggregated_time)
fig, ax = aggregated_time.plot_effects()

# %% [markdown]
# ### Event Study Aggregation
#
# To obtain event-study-type effects one can would like to aggregate $ATT(\mathrm{g}, t_\text{eval})$ over $e = t_\text{eval} - \mathrm{g}$ (respecting group size).
# As a sample oracle we will combine all `ite`'s based on group $\mathrm{g}$. As oracle values, we obtain

# %%
df["e"] = pd.to_datetime(df["t"]).values.astype("datetime64[M]") - \
    pd.to_datetime(df["d"]).values.astype("datetime64[M]")
df.groupby("e")["ite"].mean()[1:]

# %% [markdown]
# Analogously, `aggregation="eventstudy"` aggregates $\widehat{ATT}(\mathrm{g},t_\text{pre},t_\text{eval})$ based on exposure time $e = t_\text{eval} - \mathrm{g}$ (respecting group size).

# %%
aggregated_eventstudy = dml_obj.aggregate("eventstudy")
print(aggregated_eventstudy)
aggregated_eventstudy.plot_effects()

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

# %%
print(aggregated_eventstudy)

# %% [markdown]
# Here, the overall effect aggregation aggregates each effect with positive exposure

# %%
print(aggregated_eventstudy.overall_aggregation_weights)

# %% [markdown]
# If one would like to consider how the aggregated effect with $e=0$ is computed, one would have to look at the corresponding set of weights within the ``aggregation_weights`` property

# %%
# the weights for e=0 correspond to the fifth element of the aggregation weights
aggregated_eventstudy.aggregation_weights[4]

# %% [markdown]
# Taking a look at the original `dml_obj`, one can see that this combines the following estimates (only show month):
#
#  - $\widehat{ATT}(04,03,04)$
#  - $\widehat{ATT}(05,04,05)$
#  - $\widehat{ATT}(06,05,06)$

# %%
print(dml_obj.summary["coef"])

# %% [markdown]
# ## Anticipation
#
# As described in the [Model Guide](https://docs.doubleml.org/stable/guide/models.html#difference-in-differences-models-did), one can include anticipation periods $\delta>0$ by setting the `anticipation_periods` parameter.

# %% [markdown]
# ### Data with Anticipation
#
# The DGP allows to include anticipation periods via the `anticipation_periods` parameter.
# In this case the observations will be "shifted" such that units anticipate the effect earlier and the exposure effect is increased by the number of periods where the effect is anticipated.

# %%
n_obs = 4000
n_periods = 6

df_anticipation = make_did_CS2021(n_obs, dgp_type=4, n_periods=n_periods, n_pre_treat_periods=3, time_type="datetime", anticipation_periods=1)

print(df_anticipation.shape)
df_anticipation.head()


# %% [markdown]
# To visualize the anticipation, we will again plot the "oracle" values

# %%
df_anticipation["ite"] = df_anticipation["y1"] - df_anticipation["y0"]
df_anticipation["First Treated"] = df_anticipation["d"].dt.strftime("%Y-%m").fillna("Never Treated")
agg_df_anticipation = df_anticipation.groupby(["t", "First Treated"]).agg(**agg_dictionary).reset_index()
agg_df_anticipation.head()

# %% [markdown]
# One can see that the effect is already anticipated one period before the actual treatment assignment.

# %%
plot_data(agg_df_anticipation, col_name='ite')

# %% [markdown]
# Initialize a corresponding `DoubleMLPanelData` object.

# %%
dml_data_anticipation = DoubleMLPanelData(
    data=df_anticipation,
    y_col="y",
    d_cols="d",
    id_col="id",
    t_col="t",
    x_cols=["Z1", "Z2", "Z3", "Z4"],
    datetime_unit="M"
)

# %% [markdown]
# ### ATT Estimation
#
# Let us take a look at the estimation without anticipation.

# %%
dml_obj_anticipation = DoubleMLDIDMulti(dml_data_anticipation, **default_args)
dml_obj_anticipation.fit()
dml_obj_anticipation.bootstrap(n_rep_boot=5000)
dml_obj_anticipation.plot_effects()

# %% [markdown]
# The effects are obviously biased. To include anticipation periods, one can adjust the `anticipation_periods` parameter. Correspondingly, the outcome regression (and not yet treated units) are adjusted.

# %%
dml_obj_anticipation = DoubleMLDIDMulti(dml_data_anticipation, **(default_args| {"anticipation_periods": 1}))
dml_obj_anticipation.fit()
dml_obj_anticipation.bootstrap(n_rep_boot=5000)
dml_obj_anticipation.plot_effects()

# %% [markdown]
# ## Group-Time Combinations
#
# The default option `gt_combinations="standard"` includes all group time values with the specific choice of $t_\text{pre} = \min(\mathrm{g}, t_\text{eval}) - 1$ (without anticipation) which is the weakest possible parallel trend assumption.
#
# Other options are possible or only specific combinations of $(\mathrm{g},t_\text{pre},t_\text{eval})$.

# %% [markdown]
# ### All Combinations
#
# The  option `gt_combinations="all"` includes all relevant group time values with $t_\text{pre} < \min(\mathrm{g}, t_\text{eval})$, including longer parallel trend assumptions.
# This can result in multiple estimates for the same $ATT(\mathrm{g},t)$, which have slightly different assumptions (length of parallel trends).

# %%
dml_obj_all = DoubleMLDIDMulti(dml_data, **(default_args| {"gt_combinations": "all"}))
dml_obj_all.fit()
dml_obj_all.bootstrap(n_rep_boot=5000)
dml_obj_all.plot_effects()

# %% [markdown]
# ### Universal Base Period
#
# The  option `gt_combinations="universal"` set $t_\text{pre} = \mathrm{g} - \delta - 1$, corresponding to a universal/constant comparison or base period.
#
# Remark that this implies $t_\text{pre} > t_\text{eval}$ for all pre-treatment periods (accounting for anticipation). Therefore these effects do not have the same straightforward interpretation as ATT's.

# %%
dml_obj_universal = DoubleMLDIDMulti(dml_data, **(default_args| {"gt_combinations": "universal"}))
dml_obj_universal.fit()
dml_obj_universal.bootstrap(n_rep_boot=5000)
dml_obj_universal.plot_effects()

# %% [markdown]
# ### Selected Combinations
#
# Instead it is also possible to just submit a list of tuples containing $(\mathrm{g}, t_\text{pre}, t_\text{eval})$ combinations. E.g. only two combinations

# %%
gt_dict = {
    "gt_combinations": [
        (np.datetime64('2025-04'),
         np.datetime64('2025-01'),
         np.datetime64('2025-02')),
        (np.datetime64('2025-04'),
         np.datetime64('2025-02'),
         np.datetime64('2025-03')),
    ]
}

dml_obj_all = DoubleMLDIDMulti(dml_data, **(default_args| gt_dict))
dml_obj_all.fit()
dml_obj_all.bootstrap(n_rep_boot=5000)
dml_obj_all.plot_effects()
