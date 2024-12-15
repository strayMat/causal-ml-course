# %%
import numpy as np
import matplotlib.pyplot as plt
from mleco.constants import DIR2FIG
from mleco.figures import style_figs
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.RandomState(0)

n_sample = 100
data_max, data_min = 1.4, -1.4
len_data = data_max - data_min
# sort the data to make plotting easier later
data = np.sort(rng.rand(n_sample) * len_data - len_data / 2)
noise = rng.randn(n_sample) * 0.3
target = data**3 - 0.5 * data**2 + noise

# %%
import pandas as pd

full_data = pd.DataFrame({"X": data, "Y": target})
# %%
import seaborn as sns

_ = sns.scatterplot(data=full_data, x="X", y="Y", color="black", alpha=0.5)
plt.savefig(DIR2FIG / "2_linear_regression_non_linear_link.svg", bbox_inches="tight")
# %%
# X should be 2D for sklearn: (n_samples, n_features)
data = data.reshape((-1, 1))
data.shape


# %%
def fit_score_plot_regression(model, title=None):
    model.fit(data, target)
    target_predicted = model.predict(data)
    mse = mean_squared_error(target, target_predicted)
    r2 = r2_score(target, target_predicted)
    ax = sns.scatterplot(data=full_data, x="X", y="Y", color="black", alpha=0.5)
    ax.plot(data, target_predicted)
    if title is not None:
        _ = ax.set_title(title + f"\n(MSE = {mse:.2f}, R2 = {r2:.2f})")
    else:
        _ = ax.set_title(f"Mean squared error = {mse:.2f}")


# %% [markdown]
## Use a simple linear regression model
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_regression = LinearRegression()
linear_regression
# %%
fit_score_plot_regression(linear_regression, title="Simple linear regression")
plt.savefig(
    DIR2FIG / "2_linear_regression_non_linear_link_linear.svg", bbox_inches="tight"
)

# %% [markdown]
## Use a polynomial regression model
# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(
    PolynomialFeatures(3, include_bias=False), LinearRegression()
)
polynomial_regression
# %%
fit_score_plot_regression(polynomial_regression, title="Polynomial regression")
plt.savefig(
    DIR2FIG / "2_linear_regression_non_linear_link_polynomial.svg", bbox_inches="tight"
)
# %%
