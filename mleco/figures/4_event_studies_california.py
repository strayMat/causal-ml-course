# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from six import b
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

color_palette = {s: style_figs.CONTROL_COLOR for s in cigar["state"].unique()}
sns.lineplot(
    ax=ax,
    data=cigar.query("~california"),
    x="year",
    y="cigsale",
    # color=style_figs.CONTROL_COLOR,
    hue="state",
    palette=color_palette,
    alpha=0.4,
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
    cigar.query("california")["year"],
    cigar.query("california")["cigsale"],
    color=style_figs.TREATED_COLOR,
)
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=style_figs.TREATED_COLOR, lw=2, label="California"),
        plt.Line2D(
            [0],
            [0],
            color=style_figs.CONTROL_COLOR,
            lw=2,
            linestyle="--",
            label="Other States",
        ),
    ]
)
ax.set_ylim((40, 300))
plt.savefig(DIR2FIG / "scm_california_and_other_states.svg", bbox_inches="tight")
# %%
outcome_name = "cigsale"
Y = cigar.pivot(index="year", columns="state", values=outcome_name)
Y_train = Y[Y.index < 1988]
Y_control_train = Y_train.drop(columns=[3])
Y_treated_train = Y_train[3]
Y_control = Y.drop(columns=[3])
Y_treated = Y[3]
# adding more covariates
predictors = cigar.pivot(index="year", columns="state", values="retprice")
predictors_train = predictors[predictors.index < 1988]
X = pd.concat([Y_train, predictors_train], axis=0)
X_control = X.drop(columns=[3])
X_treated = X[3]
# X_test = data[data.index >= 1988].drop(columns=[3])
# %%
# Simple OLS
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression(fit_intercept=False).fit(X_control, X_treated)
# model = Lasso(alpha=10).fit(X_train, y_train)
calif_synth_lr = model.predict(Y.drop(columns=[3]))


# %%
def plot_california_vs_synthetic(data_long: pd.DataFrame, sythetic_data: pd.DataFrame):
    """
    Simple plot function to compare California and a synthetic control.

    Args:
        data_long (pd.DataFrame): _description_
        sythetic_data (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    california_data = data_long.query("california")
    california_x = california_data["year"]
    california_y = california_data["cigsale"]
    assert (
        len(california_x) == len(sythetic_data)
    ), f"Lengths do not match: California {len(california_x)} and Synthetic {len(sythetic_data)}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        california_x, california_y, label="California", color=style_figs.TREATED_COLOR
    )
    ax.plot(
        california_x,
        sythetic_data,
        label="Synthetic Control",
        color=style_figs.CONTROL_COLOR,
    )
    ax.vlines(
        x=1988,
        ymin=40,
        ymax=200,
        linestyle=":",
        lw=2,  # label="Proposition 99"
    )
    plt.ylabel("Cigarette Sales (packs/hab)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.ylim((40, 140))
    return fig, ax


plot_california_vs_synthetic(cigar, calif_synth_lr)
plt.savefig(DIR2FIG / "scm_california_vs_synth_lr.svg", bbox_inches="tight")
# %% [markdown]
# ## Synthetic Control Method without the inner optimization problem

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

model.fit(X_control, X_treated)
calif_synth_wo_v = model.predict(Y.drop(columns=[3]))
plot_california_vs_synthetic(cigar, calif_synth_wo_v)
plt.savefig(DIR2FIG / "scm_california_vs_synth_wo_v.svg", bbox_inches="tight")
# %%
from pysyncon import Dataprep, Synth

# dataprep = Dataprep(
#     foo=cigar,
#     predictors=["cigsale", "retprice"],
#     predictors_op="mean",
#     time_predictors_prior=range(1980, 1988),
#     dependent="cigsale",
#     unit_variable="state",
#     time_variable="year",
#     treatment_identifier=3,
#     controls_identifier=np.array(
#         list(set(cigar["state"].values).difference([3]))
#     ).tolist(),
#     time_optimize_ssr=range(1970, 1988),
# )
# print(dataprep)

synth = Synth()
synth.fit(
    X0=X_control,
    X1=X_treated,
    Z0=Y_control_train,
    Z1=Y_treated_train,
    optim_method="Nelder-Mead",
    optim_initial="equal",
)
# %%
print(synth.weights().T)
# predict on the full history
california_synth = Y_control @ synth.W.T
plot_california_vs_synthetic(cigar, california_synth)
plt.savefig(DIR2FIG / "scm_california_vs_synth_pysyncon.svg", bbox_inches="tight")
# %%
outcome_name = "cigsale"
Y = cigar.pivot(index="year", columns="state", values=outcome_name)
# adding more covariates
predictors = cigar.pivot(index="year", columns="state", values="retprice")
predictors_train = predictors[predictors.index < 1988]
Y_train = Y[Y.index < 1988]
X = pd.concat([Y_train, predictors_train], axis=0)
X_control = X.drop(columns=[3])
X_treated = X[3]
# %%
from tqdm import tqdm


effects = {}
for state in tqdm(Y.columns):
    model_iter = SyntheticControl()
    X_control = X.drop(columns=[state])
    X_treated = X[state]
    model_iter.fit(X_control, X_treated)

    effect = Y[state] - model_iter.predict(Y.drop(columns=[state]))
    effects[state] = effect

# %%
effect_df = pd.DataFrame(effects)
# %%
plt.figure(figsize=(10, 5))
# axes
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2, color="Black")
plt.vlines(
    x=1988,
    ymin=-50,
    ymax=100,
    linestyle=":",
    lw=2,
    color="Black",
)
# plot states
for s, effect in effect_df.items():
    if s != 3:
        plt.plot(effect, color=style_figs.CONTROL_COLOR, alpha=0.3)
plt.plot(effect_df[3], color=style_figs.TREATED_COLOR)
# formatting
plt.ylabel("Effect Estimate")
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=style_figs.TREATED_COLOR, lw=2, label="California"),
        plt.Line2D(
            [0],
            [0],
            color=style_figs.CONTROL_COLOR,
            lw=2,
            linestyle="--",
            label="Other States",
        ),
    ]
)
plt.grid(alpha=0.3)
plt.savefig(DIR2FIG / "scm_placebo.svg", bbox_inches="tight")
