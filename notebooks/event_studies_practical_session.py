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
# # Plan of the practical session:
#  - Application of the ITS method with a simple SARIMA model with the `statsmodels` and a more elaborated model with `causalimpact` packages.
#  - Application of the synthetic control using the `pysyncon` package.
#
# # Data
# - STEP 1. Replicating the original CITS analyses by [Humphreys et al. (2017)](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2582988) using four comparison states (New York, New Jersey, Ohio, Virginia).
# - STEP 2. Extending the original CITS analyses by [Humphreys et al. (2017)](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2582988) using synthetic control methods (15 comparison states in the donor pool).
# The code is inspired by the analyses in the original papers from [Degli Esposti et al., 2020](https://academic.oup.com/ije/article/49/6/2010/5917161#supplementary-data). It uses the data from (Bonander et al., 2021)[https://academic.oup.com/aje/article-abstract/190/12/2700/6336907].

#  %%
from httpx import post
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skrub import TableReport
from datetime import datetime
from sklearn.impute import SimpleImputer


# %% [markdown]
# # STEP 1: Synthetic control method

# First we set some constants.
# %%
# set some constants
URL2DATA = "https://raw.githubusercontent.com/strayMat/causal-ml-course/refs/heads/main/data/homicides_data.csv"

TREATED_COLOR = plt.get_cmap("tab10")(0)  # Default matplotlib blue
CONTROL_COLOR = plt.get_cmap("tab10")(1)  # Default matplotlib orange

COL_TREATMENT = "treatdummy"
COL_TREATED_UNIT = "Case"
TREATMENT_DATE = datetime(2005, 10, 1)
COL_TARGET = "HomicideRates"
RANDOM_SEED = 93572
# %% [markdown]
# Load the data and add a column with the Date format from the year and month columns homicides.
# %%
homicides = pd.read_csv(URL2DATA, index_col=0)
# ["Month.string"] = homicides["Month.code"].apply(lambda x: f"{x:02d}")
homicides["Date"] = pd.to_datetime(
    homicides["Year"].astype(str) + "-" + homicides["Month.string"], format="%Y-%m"
)


# TableReport(homicides)
# %% [markdown]
# plotting utils
def plot_observed_data(
    data: pd.DataFrame,
    target_column: str = COL_TARGET,
    target_unit_name: str = "Florida",
    plot_all_controls: bool = False,
):
    """
    Plotting the data by treatment status.

    Args:
        data (pd.DataFrame): Observed data to plot
        plot_all_controls (bool): Whether to plot all control units or just the mean and sd for all control units.
    """

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # plot treated unit
    data_treated = data[data["State"] == target_unit_name]
    sns.lineplot(
        data=data_treated,
        x="Date",
        y=target_column,
        ax=ax,
        color=TREATED_COLOR,
        # marker="o",
        label=target_unit_name,
    )
    data_controls = data[data["State"] != target_unit_name]
    color_palette = {s: CONTROL_COLOR for s in data_controls["State"].unique()}
    if plot_all_controls:
        sns.lineplot(
            ax=ax,
            data=data_controls,
            x="Date",
            y=target_column,
            hue="State",
            palette=color_palette,
            alpha=0.7,
            # marker="o",
            linestyle="--",
        )
    # plot mean and sd for controls
    else:
        mean_controls = data_controls.groupby("Date")[COL_TARGET].mean()
        sd_controls = data_controls.groupby("Date")[COL_TARGET].std()
        ax.fill_between(
            mean_controls.index,
            mean_controls - sd_controls,
            mean_controls + sd_controls,
            color=CONTROL_COLOR,
            alpha=0.3,
        )
        ax.plot(mean_controls, color=CONTROL_COLOR, label="Control States")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color=TREATED_COLOR, lw=2, label=target_unit_name),
            plt.Line2D([0], [0], color=CONTROL_COLOR, lw=2, label="Control States"),
        ]
    )
    # plot treatment date
    ax.axvline(TREATMENT_DATE, color="black", linestyle="--")
    return fig, ax


def plot_data_with_prediction(data: pd.DataFrame, y_pred, lower_ci=None, upper_ci=None):
    """
    Plotting a prediction with confidence intervals on top of the observed data.

    Args:
        data (pd.DataFrame): Observed data
        y_pred (_type_): Mean of the predicted values
        lower_ci (_type_): Lower bound of the confidence interval for the predicted values
        upper_ci (_type_): Upper bound of the confidence interval for the predicted values

    Returns:
        _type_: _description_
    """
    fig, ax = plot_observed_data(data)

    dates_to_predict = np.sort(data["Date"].unique())
    ax.plot(
        dates_to_predict,
        y_pred,
        color="black",
        linestyle="-.",
        label="Predicted Florida",
    )
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            dates_to_predict,
            lower_ci,
            upper_ci,
            color="black",
            alpha=0.3,
        )
    return fig, ax


# %% [markdown]
# ## Plot the subset of states that we want to include in the analysis.
# %% [markdown]
# %%
mask_subset_states = homicides["State"].isin(
    ["Florida", "New York", "New Jersey", "Ohio", "Virginia"]
)
homicides_subset = homicides[mask_subset_states]
plot_observed_data(homicides[mask_subset_states], plot_all_controls=True)
# %% [markdown]
# TODO: Does the data seems consistent with the convex-hull assumption?
# Find a subsest for which the convex-hull assumption do not seems to hold.
# TODO: plot the data that you have chosen with only the mean and standard deviation of the control units.
# %%
homicides_subset = homicides
plot_observed_data(homicides_subset, plot_all_controls=False)
print(homicides["State"].unique())
# %%
from pysyncon import Synth

mask_treated = homicides_subset["State"] == "Florida"
mask_pre_treatment = homicides_subset["Date"] < TREATMENT_DATE

control_units = homicides_subset[~mask_treated]
treated_unit = homicides_subset[mask_treated]
Y_control = control_units.pivot(index="Date", columns="State", values=[COL_TARGET])
Y_treated = treated_unit[COL_TARGET]

Y_pretreatment_control = control_units.loc[
    control_units["Date"] < TREATMENT_DATE
].pivot(index="Date", columns="State", values=[COL_TARGET])

Y_pretreatment_treated = treated_unit.loc[
    treated_unit["Date"] < TREATMENT_DATE, COL_TARGET
]


temporal_covariates = [COL_TARGET]  # , "Unemployment_adj"]
static_covariates = [
    "Burglary.rate",
    "Population",
    "Personal.income.per.capita..dollars.",
    "Paid.Hunting.License.Holders",
]

pretreatment_controls = homicides_subset[(~mask_treated) & mask_pre_treatment]
pretreatment_treated = homicides_subset[mask_treated & mask_pre_treatment]

control_covariates_list = []
treated_covariates_list = []
for cov in temporal_covariates:
    control_covariates_list.append(
        pretreatment_controls.pivot(index="Date", columns="State", values=cov)
    )

    treated_covariates_list.append(
        pretreatment_treated.pivot(
            index="Date", columns="State", values=temporal_covariates
        ).iloc[:, 0]
    )
control_static_covariates_list = []
treated_static_covariates_list = []
for cov in static_covariates:
    control_static_covariates_list.append(
        control_units.pivot(index="Date", columns="State", values=cov).iloc[0, :].T
    )
    treated_static_covariates_list.append(
        treated_unit.pivot(index="Date", columns="State", values=cov).iloc[0, :]
    )
control_static_covariates = pd.concat(control_static_covariates_list, axis=1).T
treated_static_covariates = pd.concat(treated_static_covariates_list, axis=0)
control_covariates = pd.concat(
    (pd.concat(control_covariates_list, axis=0), control_static_covariates), axis=0
)
treated_covariates = pd.concat(
    (pd.concat(treated_covariates_list, axis=0), treated_static_covariates), axis=0
)
# %%
synth = Synth()
synth.fit(
    dataprep=None,
    X0=Y_pretreatment_control,
    X1=Y_pretreatment_treated,
    Z0=control_covariates,
    Z1=treated_covariates,
    optim_method="L-BFGS-B",  # "Nelder-Mead",
    # optim_initial="equal",
)
# %%
y_pred = Y_control @ synth.W.T
plot_data_with_prediction(homicides_subset, y_pred)

# %%
# STEP 1: ITS with AR and SARIMAx models. Try different ARIMA models
# %%
states_subset_for_its = ["Florida", "New York", "New Jersey", "Ohio", "Virginia"]
homicides_subset_for_its = homicides[homicides["State"].isin(states_subset_for_its)]
print(homicides_subset_for_its["State"].unique())

# Plotting the data
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.lineplot(
    data=homicides_subset_for_its,
    x="Date",
    y=COL_TARGET,
    hue="State",
    ax=ax,
    marker="o",
)
plt.show()


# Define subsets of the dataset of interest
T = homicides_subset_for_its["Date"].unique().shape[0]
pretreatment_data = homicides_subset_for_its[
    homicides_subset_for_its["Date"] < TREATMENT_DATE
]
posttreatment_data = homicides_subset_for_its[
    homicides_subset_for_its["Date"] >= TREATMENT_DATE
]
pretreatment_treated = pretreatment_data[pretreatment_data[COL_TREATED_UNIT] == 1]
posttreatment_treated = posttreatment_data[posttreatment_data[COL_TREATED_UNIT] == 1]
# some interesting predictors
predictor_names = [
    "Unemployment_adj",
]
# %%
from statsmodels.tsa.api import AutoReg

autoreg = AutoReg(
    endog=pretreatment_treated[COL_TARGET],
    lags=1,
    seasonal=True,
    period=12,
)
auto_reg_fit = autoreg.fit()
print(auto_reg_fit.summary())
auto_reg_pred = auto_reg_fit.get_prediction(start=1, end=T)
lower_ci = auto_reg_pred.conf_int().iloc[:, 0]
upper_ci = auto_reg_pred.conf_int().iloc[:, 1]
y_pred = auto_reg_pred.predicted_mean
_, ax = plot_data_with_prediction(homicides_subset_for_its, y_pred, lower_ci, upper_ci)
# %%
# Using a sarimax model
# Prepare predictors
pretreatment_treated_predictors = SimpleImputer().fit_transform(
    pretreatment_treated[predictor_names]
)
posttreatment_treated_predictors = SimpleImputer().fit_transform(
    posttreatment_treated[predictor_names]
)
treated_unit_predictors = SimpleImputer().fit_transform(
    homicides_subset_for_its.loc[
        homicides_subset_for_its[COL_TREATED_UNIT] == 1, predictor_names
    ]
)

from statsmodels.tsa.api import SARIMAX

sarimax = SARIMAX(
    endog=pretreatment_treated[COL_TARGET],
    order=(1, 0, 2),
    seasonal_order=(0, 0, 0, 12),
)

sarimax_fit = sarimax.fit()
print(sarimax_fit.summary())
sarimax_pred = sarimax_fit.get_prediction(start=1, end=T)
lower_ci = sarimax_pred.conf_int().iloc[:, 0]
upper_ci = sarimax_pred.conf_int().iloc[:, 1]
y_pred = sarimax_pred.predicted_mean
_, ax = plot_data_with_prediction(homicides_subset_for_its, y_pred, lower_ci, upper_ci)
ax.set_ylim((0, 300))
# %%


# %%
# TODO: why is this not wrking?
# Compare with best sarimax model with predictors
from pmdarima import auto_arima


bic_best_model = auto_arima(
    pretreatment_treated[COL_TARGET],
    X=pretreatment_treated_predictors,
    seasonal=True,
    stepwise=True,
    max_p=3,
    max_d=2,
    max_q=3,
    m=12,
    suppress_warnings=True,
    n_jobs=-1,
    # random_state=RANDOM_SEED,
    # information_criterion="bic",
)


y_pred, conf_int = bic_best_model.predict(
    n_periods=T, X=treated_unit_predictors, return_conf_int=True
)
plot_data_with_prediction(
    homicides_subset_for_its, y_pred, conf_int[:, 0], conf_int[:, 1]
)
bic_best_model
# %%
# Compare with a SARIMAX model (should be close from causal impact)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%
# Necessary for causal impact
# !pip install pandas==1.5.2
# %%
# Make the analysis super easy with causal impact
from causalimpact import CausalImpact

pre_period = [
    pd.to_datetime(pretreatment_data["Date"].min()),
    pd.to_datetime(TREATMENT_DATE),
]
post_period = [
    pd.to_datetime(TREATMENT_DATE + pd.DateOffset(months=1)),
    pd.to_datetime(homicides_subset_for_its["Date"].max()),
]
data_for_causal_impact = homicides_subset_for_its.loc[
    homicides_subset_for_its[COL_TREATED_UNIT] == 1,
    ["Date", COL_TARGET, *predictor_names],
]
data_for_causal_impact["Date"] = pd.to_datetime(data_for_causal_impact["Date"])
data_for_causal_impact.set_index("Date", inplace=True)
data_for_causal_impact.sort_index(inplace=True)

impact = CausalImpact(data_for_causal_impact, pre_period, post_period)
# %%
print(impact.summary())
print(impact.summary(output="report"))
impact.plot()
# %%)
