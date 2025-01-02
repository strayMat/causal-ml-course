# %%
from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mleco.constants import DIR2FIG, DIR2DATA
from mleco.figures import style_figs

# %%
smoking_url = "https://raw.githubusercontent.com/matheusfacure/python-causality-handbook/refs/heads/master/causal-inference-for-the-brave-and-true/data/smoking.csv"

path2data = DIR2DATA / "california_smoking.csv"
if not path2data.exists():
    cigar = pd.read_csv(smoking_url).drop(columns=["lnincome", "beer", "age15to24"])
    cigar.to_csv(path2data, index=False)
else:
    cigar = pd.read_csv(path2data)
cigar.query("california").head()
# %%
# plot cigarette sales in California vs other states
fig, ax = plt.subplots(figsize=(10, 5))

pivoted_df = (
    cigar.assign(california=np.where(cigar["california"], "California", "Other States"))
    .groupby(["year", "california"])["cigsale"]
    .mean()
    .reset_index()
    .pivot(index="year", columns="california", values="cigsale")
)
ax.plot(pivoted_df, label=["California", "Other States"])


plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales (packs/hab)")
# plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend()
plt.savefig(DIR2FIG / "scm_california_vs_other_states.svg", bbox_inches="tight")

# %%
# plot cigarette sales in California and other states
fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(
    ax=ax,
    data=cigar,
    x="year",
    y="cigsale",
    hue="state",
    palette=sns.color_palette("husl", 39),
    alpha=0.5,
    linestyle="--",
)

plt.vlines(
    x=1988,
    ymin=0,
    ymax=350,
    linestyle=":",
    lw=2,
)
plt.ylabel("Cigarette Sales (packs/hab)")
# plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend().remove()
ax.plot(
    cigar.loc[cigar["california"], ["year", "cigsale"]].set_index("year"),
    color=style_figs.TREATED_COLOR,
    label="California",
)
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=style_figs.TREATED_COLOR, lw=2, label="California"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--", label="Other States"),
    ]
)
ax.set_ylim((40, 300))
plt.savefig(DIR2FIG / "scm_california_and_other_states.svg", bbox_inches="tight")
# %%
# Simple OLS
cigsales_by_year_state = cigar.pivot(index="year", columns="state", values="cigsale")
cigsales_with_covariates = pd.concat(
    [
        cigsales_by_year_state,
        cigar.pivot(index="year", columns="state", values="retprice"),
    ],
    axis=0,
)
data = cigsales_with_covariates
train = data[data.index < 1988]
X_train = train.drop(columns=[3])
y_train = train[3]
X_test = data[data.index >= 1988].drop(columns=[3])
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
# model = Lasso(alpha=10).fit(X_train, y_train)
calif_synth_lr = model.predict(cigsales_by_year_state.drop(columns=[3]))
# %%
plt.figure(figsize=(8, 6))
plt.plot(
    cigar.query("california")["year"],
    cigar.query("california")["cigsale"],
    label="California",
)
plt.plot(cigar.query("california")["year"], calif_synth_lr, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales (packs/hab)")
plt.legend()
plt.savefig(DIR2FIG / "scm_california_vs_synth_lr.svg", bbox_inches="tight")
# %%
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp


class SyntheticControl(BaseEstimator, RegressorMixin):
    """
    Synthetic Control Method without the inner optimization problem.
    """

    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        w = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ w - y))

        constraints = [cp.sum(w) == 1, w >= 0]

        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        self.X_ = X
        self.y_ = y
        self.w_ = w.value

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.w_


# %%

model = SyntheticControl()

model.fit(X_train, y_train)
# %%
