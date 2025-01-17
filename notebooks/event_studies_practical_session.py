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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skrub import TableReport
from datetime import datetime

# %%
URL2DATA = "https://raw.githubusercontent.com/strayMat/causal-ml-course/refs/heads/main/data/homicides_data.csv"

TREATED_COLOR = plt.get_cmap("tab10")(0)  # Default matplotlib blue
CONTROL_COLOR = plt.get_cmap("tab10")(1)  # Default matplotlib orange

COL_TREATMENT = "treatdummy"
COL_TREATED_UNIT = "Case"
TREATMENT_DATE = datetime(2005, 10, 1)
# %%
# Load the data 
homicides = pd.read_csv(URL2DATA, index_col=0)
# add a column with the Date format from the year and month columns
homicides["Month.string"] = homicides["Month.code"].apply(lambda x: f"{x:02d}")
homicides["Date"] = pd.to_datetime(homicides["Year"].astype(str)+"-"+homicides["Month.string"], format="%Y-%m")
TableReport(homicides)
# %%
# %%
states_subset_for_its = ["Florida","New York", "New Jersey", "Ohio", "Virginia"]
homicides_subset_for_its = homicides[homicides["State"].isin(states_subset_for_its)]
print(homicides_subset_for_its["State"].unique())
# %%
# Plotting the data
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.lineplot(data=homicides_subset_for_its, x="Date", y="HomicideRates", hue="State", ax=ax, marker="o")
plt.show()
# %%
# Plotting the data by treatment status
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.lineplot(data=homicides_subset_for_its, x="Date", y="HomicideRates", hue=COL_TREATED_UNIT, ax=ax, marker="o", palette={0: CONTROL_COLOR, 1: TREATED_COLOR})
# plot treatment date
ax.axvline(TREATMENT_DATE, color="black", linestyle="--")
plt.show()
# %% 

pretreatment_data = homicides_subset_for_its[homicides_subset_for_its["Date"] < TREATMENT_DATE]
posttreatment_data = homicides_subset_for_its[homicides_subset_for_its["Date"] >= TREATMENT_DATE]
pretreatment_treated = pretreatment_data[pretreatment_data[COL_TREATMENT] == 1]

# %%
# STEP 1: ITS with SARIMA model
# %%
from statsmodels.tsa.arima.model import ARIMA
arima = ARIMA(pretreatment_treated["HomicideRates"], order=(1, 0, 0))
arima_fit = arima.fit()
print(arima_fit.summary())
y_pred_w_ci = arima_fit.get_prediction(
    start=0, end=(N_OBS - 1)
)
ci = y_pred_w_ci.conf_int()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# %%
# Compare with a SARIMAX model (should be close from causal impact)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %% 
# Make the analysis super easy with causal impact

# %% 
# STEP 2: Synthetic control method