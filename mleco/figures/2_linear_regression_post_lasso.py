# %%
from numpy import ndarray
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from mleco.figures import style_figs
from mleco.constants import DIR2FIG

RANDOM_STATE = 1
rg = check_random_state(RANDOM_STATE)
# %%
# # Approximate sparsity model
n = 1000
p = 300
noise_var = 1
beta_0 = [1 / j**2 for j in range(1, p + 1)]
X = rg.normal(0, 1, (n, p))
epsilon = rg.normal(0, noise_var, n)
y = X @ beta_0 + epsilon


# %% [markdown]
def plot_coeffs(beta, label=None, ax=None):
    """utility function to plot coefficients.

    Args:
        beta (_type_): _description_
        label (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        beta,
        marker="o",
        markersize=10,
        label=label,
        markerfacecolor="white",
        linewidth=2,
    )
    ax.set_xlim(-1, 20)
    ax.set_xlabel("Feature index j")
    ax.set_ylabel("j-th largest true coefficient")
    return ax


ax = plot_coeffs(beta_0, label="True coefficients")
plt.legend()
plt.savefig(DIR2FIG / "approximate_sparse_true_coeffs.svg", bbox_inches="tight")
# %% [markdown]
# Fit a linear regression model
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
linear = LinearRegression()
pipeline = make_pipeline(scaler, linear)
ols_coeffs = pipeline.fit(X, y).named_steps["linearregression"].coef_
ax = plot_coeffs(beta_0, label="True coefficients")
plot_coeffs(ols_coeffs, ax=ax, label="OLS coefficients")
plt.legend()
plt.savefig(DIR2FIG / "approximate_sparse_ols_coeffs.svg", bbox_inches="tight")
# %% [markdown]
# Fit a Lasso model
scaler = StandardScaler()
lasso = Lasso(alpha=0.1, random_state=RANDOM_STATE)
params_distributions = {"lasso__alpha": [0.1, 0.5, 1, 5, 10, 10e2]}
pipeline = make_pipeline(scaler, lasso)

cross_val_lasso = RandomizedSearchCV(
    pipeline,
    param_distributions=params_distributions,
    cv=5,
    n_iter=6,
    random_state=RANDOM_STATE,
)
cross_val_lasso.fit(X, y)
lasso_coeffs = cross_val_lasso.best_estimator_.named_steps["lasso"].coef_
ax = plot_coeffs(beta_0, label="True coefficients")
plot_coeffs(ols_coeffs, ax=ax, label="OLS coefficients")
plot_coeffs(lasso_coeffs, ax=ax, label="Lasso coefficients")
plt.legend()
plt.savefig(DIR2FIG / "approximate_sparse_lasso_coeffs.svg", bbox_inches="tight")


# %% [markdown]
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel


# # Post-lasso estimation
class PostLasso(LinearModel):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

        self.lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=self.warm_start,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
        )
        self.ols = LinearRegression()

    def fit(self, X, y):
        self.lasso.fit(X, y)
        active_set = self.lasso.coef_ != 0
        self.active_set = active_set
        self.ols.fit(X[:, active_set], y)
        return self

    def predict(self, X):
        return self.ols.predict(X[:, self.active_set])


scaler = StandardScaler()
postlasso = PostLasso(alpha=0.1, random_state=RANDOM_STATE)
pipeline = make_pipeline(scaler, postlasso)

# %%
params_distributions_postlasso = {
    "postlasso__alpha": params_distributions["lasso__alpha"]
}
cross_val_postlasso = RandomizedSearchCV(
    pipeline,
    param_distributions=params_distributions_postlasso,
    cv=5,
    n_iter=6,
    random_state=RANDOM_STATE,
    scoring="neg_mean_squared_error",
)
cross_val_postlasso.fit(X, y)
postlasso_active_coeffs = cross_val_postlasso.best_estimator_.named_steps[
    "postlasso"
].ols.coef_
postlasso_coeffs = np.concat(
    [postlasso_active_coeffs, np.zeros(p - len(postlasso_active_coeffs))]
)

# %%
ax = plot_coeffs(beta_0, label="True coefficients")
plot_coeffs(ols_coeffs, ax=ax, label="OLS coefficients")
plot_coeffs(lasso_coeffs, ax=ax, label="Lasso coefficients")
plot_coeffs(postlasso_coeffs, ax=ax, label="Post-Lasso coefficients")
plt.legend()
plt.savefig(DIR2FIG / "approximate_sparse_postlasso_coeffs.svg", bbox_inches="tight")
# %%
