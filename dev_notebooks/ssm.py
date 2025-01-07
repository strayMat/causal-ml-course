# %% 
import numpy as np
# %% [markdown]
# A state space model is a probabilistic model for a dynamical system in which the system is assumed to be governed by a latent state that evolves over time. 
# It is modeled by two equations: 
# 1. The outcome (or observation) equation  which describes how the observed data is generated from the latent state: 
# $$y_t = Z_t ^T \alpha_t + \varepsilon_t$$
# 2. The state equation, which describes how the latent state evolves over time. 
# $$\alpha_{t+1} = T_t \alpha_t + R_t \eta_t$$

## Example of state space models: from statsmodels 
# %%  
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

plt.rc("figure", figsize=(10, 5))
plt.rc("font", size=15)
TREATED_COLOR = plt.get_cmap("tab10")(0)  # Default matplotlib blue
CONTROL_COLOR = plt.get_cmap("tab10")(1)  # Default matplotlib orange
# %% 
def gen_data_for_model1(nobs, t_intervention):
    rs = np.random.RandomState(seed=93572)
    d = 5
    var_eps = 0.1
    var_coeff_a = 0.01
    var_coeff_e = 0.5
    seasonality_amplitude = 5
    seasonality_period = 12
    effect_half_life = 50
    effect_magnitude = 20
    
    alpha_t = rs.uniform(size=nobs)
    eta_t = rs.uniform(size=nobs)
    eps = rs.normal(scale=var_eps ** 0.5, size=nobs)

    beta_a = np.cumsum(rs.normal(size=nobs, scale=var_coeff_a ** 0.5))
    beta_e = np.cumsum(rs.normal(size=nobs, scale=var_coeff_e ** 0.5))
    # adding seasonality component
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(nobs) / seasonality_period)
    # assembling state space model for the untreated series
    y_untreated = d + beta_a * alpha_t + beta_e * eta_t + eps + seasonality
    # building a control series and a treated series
    var_control = 0.01
    x_control = y_untreated*2 + 5 + rs.normal(scale=var_control ** 0.5, size=nobs)
    y_treated = y_untreated.copy()
    y_treated[t_intervention:] = y_treated[t_intervention:] + effect_magnitude*(np.exp(-np.arange(nobs - t_intervention)/effect_half_life))
    return y_untreated, y_treated, x_control, alpha_t, eta_t, beta_a, beta_e

T_INTERVENTION = 50
N_OBS = 200
y_0, y_1, x_control, alpha_t, eta_t, beta_a, beta_e = gen_data_for_model1(t_intervention=T_INTERVENTION, nobs=N_OBS)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
plt.legend()

y_1_pretreatment = y_1[:T_INTERVENTION]
x_control_pretreatment = x_control[:T_INTERVENTION]
# %% 
from statsmodels.tsa.api import acf, graphics, pacf, SARIMAX
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
# %% [markdown]
# # SARIMA models 
# SARIMA are a subset of state space models that are commonly used for time series forecasting. They are well suited for univariate time series without exogenous variables. They force the time series to be stationnary.  
# 
# They are defined by four parameters:
# 1. p: the number of autoregressive terms
# 2. d: the number of differences needed to make the time series stationary (integration terms)
# 3. q: the number of moving average terms
#

# ## ARIMA 
# 
# %% 

# %% [markdown]
# ## SARIMAX 
# 
# SARIMAX models are a generalozatoo, of 

# ## Fit a SARIMAX model to the pretreatment data without any exogenous/control variables
mod = SARIMAX(y_1_pretreatment, order=(5, 0, 4), seasonal_order=(0, 0, 0, 12))
res = mod.fit()
print(res.summary())
y_pred_w_ci = res.get_prediction(exog=x_control[T_INTERVENTION:], start=0, end=(N_OBS-1))
ci = y_pred_w_ci.conf_int()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(y_pred_w_ci.predicted_mean, label=r"Predicted $y_t(0)$", color="black", linestyle="--")
ax.fill_between(np.arange(0, N_OBS), ci[:, 0], ci[:, 1], color="black", alpha=0.5)
plt.legend()
# %% 
mod = SARIMAX(y_1_pretreatment, exog=x_control_pretreatment, order=(5, 0, 0), seasonal_order=(0, 0, 0, 12))
res = mod.fit()
print(res.summary())
# %%
y_pred_w_ci = res.get_prediction(exog=x_control[T_INTERVENTION:], start=0, end=(N_OBS-1))
ci = y_pred_w_ci.conf_int()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(y_pred_w_ci.predicted_mean, label=r"Predicted $y_t(0)$", color="black", linestyle="--")
ax.fill_between(np.arange(0, N_OBS), ci[:, 0], ci[:, 1], color="black", alpha=0.5)
plt.legend()
# %% 