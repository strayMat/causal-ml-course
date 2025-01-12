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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

plt.rc("figure", figsize=(10, 5))
plt.rc("font", size=15)
TREATED_COLOR = plt.get_cmap("tab10")(0)  # Default matplotlib blue
CONTROL_COLOR = plt.get_cmap("tab10")(1)  # Default matplotlib orange
RANDOM_SEED = 93572


# %% [markdown]
# ## Simulated data for a simple state space model
# %%
def gen_data_for_model1(nobs, t_intervention, random_seed=RANDOM_SEED):
    rs = np.random.RandomState(seed=RANDOM_SEED)
    d = 5
    var_eps = 1
    var_coeff_a = 0.05
    var_coeff_e = 0.5
    seasonality_amplitude = 5
    seasonality_period = 12
    effect_half_life = 50
    effect_magnitude = 20

    alpha_t = rs.uniform(size=nobs)
    eta_t = rs.uniform(size=nobs)
    eps = rs.normal(scale=var_eps**0.5, size=nobs)

    beta_a = np.cumsum(rs.normal(size=nobs, scale=var_coeff_a**0.5))
    beta_e = np.cumsum(rs.normal(size=nobs, scale=var_coeff_e**0.5))
    # adding seasonality component
    seasonality = seasonality_amplitude * np.sin(
        2 * np.pi * np.arange(nobs) / seasonality_period
    )
    # assembling state space model for the untreated series
    # building a control series and a treated series
    var_control = 0.01
    x_control = (beta_a * alpha_t) ** 2 + rs.normal(scale=var_control**0.5, size=nobs)
    y_untreated = d + x_control + 5 + beta_e * eta_t + eps + seasonality
    y_treated = y_untreated.copy()
    y_treated[t_intervention:] = y_treated[t_intervention:] + effect_magnitude * (
        np.exp(-np.arange(nobs - t_intervention) / effect_half_life)
    )
    return y_untreated, y_treated, x_control, alpha_t, eta_t, beta_a, beta_e


T_INTERVENTION = 50
N_OBS = 200
y_0, y_1, x_control, alpha_t, eta_t, beta_a, beta_e = gen_data_for_model1(
    t_intervention=T_INTERVENTION, nobs=N_OBS
)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
plt.legend()

y_1_pretreatment = y_1[:T_INTERVENTION]
x_control_pretreatment = x_control[:T_INTERVENTION]
# %%
from statsmodels.tsa.api import SARIMAX
# %% [markdown]
# # SARIMA models with statsmodels
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

# ## Fit a SARIMAX model to the pretreatment data without any exogenous/control variables
mod = SARIMAX(y_1_pretreatment, order=(5, 0, 4), seasonal_order=(0, 0, 0, 12))
res = mod.fit()
print(res.summary())
y_pred_w_ci = res.get_prediction(
    exog=x_control[T_INTERVENTION:], start=0, end=(N_OBS - 1)
)
ci = y_pred_w_ci.conf_int()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(
    y_pred_w_ci.predicted_mean,
    label=r"Predicted $y_t(0)$",
    color="black",
    linestyle="--",
)
ax.fill_between(np.arange(0, N_OBS), ci[:, 0], ci[:, 1], color="black", alpha=0.5)
plt.legend()
# %% [markdown]
# ## Fit a SARIMAX model to the pretreatment data with the exogenous/control variable.
# %%
mod = SARIMAX(
    y_1_pretreatment,
    exog=x_control_pretreatment,
    order=(5, 0, 4),
    seasonal_order=(0, 0, 0, 12),
)
res = mod.fit()
print(res.summary())

y_pred_w_ci = res.get_prediction(
    exog=x_control[T_INTERVENTION:], start=0, end=(N_OBS - 1)
)
ci = y_pred_w_ci.conf_int()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(
    y_pred_w_ci.predicted_mean,
    label=r"Predicted $y_t(0)$",
    color="black",
    linestyle="--",
)
ax.fill_between(np.arange(0, N_OBS), ci[:, 0], ci[:, 1], color="black", alpha=0.5)
plt.legend()
# %% [markdown]
# # Selection of the best ARIMA model by information criteria

# Typically, we would minimze the Bayesian Information Criterion (BIC) or the Akaike Information Criterion (AIC) to select the best model. These are penalized version of the likelihood of the model that take into account the number of parameters in the model. The idea is to find the model that best fits the data with the fewest parameters.
#
# Let $\hat{L}$ be the empirical likelihood of the model, k the number of parameters and n the number of observations, these criteria are defined as:
# $$BIC = -2 \ln(\hat{L}) + k \ln(n)$$
# $$AIC = -2 \ln(\hat{L}) + 2k$$
# %%
from pmdarima import auto_arima

bic_best_model = auto_arima(
    y_1_pretreatment.reshape(-1, 1),
    X=x_control_pretreatment.reshape(-1, 1),
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    max_p=3,
    max_d=2,
    max_q=3,
    max_P=3,
    max_D=2,
    max_Q=3,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)
# %%
y_pred, conf_int = bic_best_model.predict(
    n_periods=len(x_control), X=x_control.reshape(-1, 1), return_conf_int=True
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(
    y_pred,
    label=r"Predicted $y_t(0)$",
    color="black",
    linestyle="--",
)
ax.fill_between(
    np.arange(0, N_OBS), conf_int[:, 0], conf_int[:, 1], color="black", alpha=0.5
)
plt.legend()
# %% [markdown]
# ## Cross-validation of a SARIMAX model

# FIXME: I dont understand why the cross-validation is not working.
# %%
from pmdarima import ARIMA
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

est = ARIMA(
    order=(0, 0, 0), seasonal_order=(0, 0, 0, 12), seasonal=True, suppress_warnings=True
)
params_grid = ParameterGrid(
    {
        # "p": np.arange(0, 3),
        # "d": [0, 1, 2],
        # "q": np.arange(0, 3),
        "P": np.arange(0, 3),
        "D": [0, 1, 2],
        "Q": np.arange(0, 3),
    }
)
tscv = TimeSeriesSplit(n_splits=2)
# %%
from joblib import Parallel, delayed


def p_cross_val_score(est, X, y, scoring, cv, order):
    # order_ = (order["p"], order["d"], order["q"])
    s_order_ = (order["P"], order["D"], order["Q"], 12)
    est_ = clone(est)
    est_.set_params(**{"seasonal_order": s_order_})  # , "order": order_
    train_test_splits = list(cv.split(y))
    cv_results = []
    try:
        for train_idx, test_idx in train_test_splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            est_.fit(y_train, X_train)
            y_pred = est_.predict(n_periods=len(X_test), X=X_test)
            cv_results.append(scoring(y_test, y_pred))
    except ValueError:
        cv_results = [np.nan]
    return cv_results


# randomized search
n_iter = 100
rs = check_random_state(RANDOM_SEED)
if n_iter > len(list(params_grid)):
    rs_grid = params_grid
else:
    rs_grid = rs.choice(list(params_grid), size=n_iter, replace=False)
cv_results = Parallel(n_jobs=-1)(
    delayed(p_cross_val_score)(
        est,
        y_1_pretreatment.reshape(-1, 1),
        x_control_pretreatment.reshape(-1, 1),
        mean_squared_error,
        tscv,
        order,
    )
    for order in rs_grid
)
cv_results_df = pd.DataFrame({"cv_results": cv_results})
# cv_results_df["order"] = [(orders["p"], orders["d"], orders["q"]) for orders in rs_grid]
cv_results_df["seasonnality_order"] = [
    (orders["P"], orders["D"], orders["Q"]) for orders in rs_grid
]
cv_results_df["mean_cv_results"] = cv_results_df["cv_results"].apply(np.mean)
best_cv_result = cv_results_df.sort_values("mean_cv_results")
# %%
best_cv_arima = ARIMA(
    # order=best_cv_result["order"].values[0],
    order=(0, 0, 0),
    seasonal_order=(*best_cv_result["seasonnality_order"].values[0], 12),
    suppress_warnings=True,
    seasonal=True,
)
best_cv_arima.fit(
    y_1_pretreatment.reshape(-1, 1), x_control_pretreatment.reshape(-1, 1)
)
print(best_cv_arima.summary())
# plotting the results
y_pred, conf_int = best_cv_arima.predict(
    n_periods=len(x_control), X=x_control.reshape(-1, 1), return_conf_int=True
)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_1, label=r"Observed $y_t(1)$", color=TREATED_COLOR)
ax.plot(y_0, label=r"Counterfactual $y_t(0)$", color=CONTROL_COLOR)
ax.plot(
    y_pred,
    label=r"Predicted $y_t(0)$",
    color="black",
    linestyle="--",
)
ax.fill_between(
    np.arange(0, N_OBS), conf_int[:, 0], conf_int[:, 1], color="black", alpha=0.5
)
plt.legend()
