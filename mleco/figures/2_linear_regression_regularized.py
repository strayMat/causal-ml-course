# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from mleco.figures import style_figs
from mleco.constants import DIR2FIG

# %%
rng = np.random.RandomState(42)
x_all = rng.randn(150) * 5
y_all = 1 * x_all + 5 * rng.randn(len(x_all))
X_all = x_all[:, None]
xlims = x_all.min() - 2, x_all.max() + 2
ylims = y_all.min() - 2, y_all.max() + 2

X_test = np.linspace(x_all.min(), x_all.max(), 100)[:, None]

train_size = 5
colors = plt.cm.tab10(np.arange(10))

training_sets = []
for i, color in zip(range(6), colors):
    X_train, _, y_train, _ = train_test_split(
        X_all, y_all, train_size=train_size, random_state=rng
    )
    training_sets.append((X_train, y_train, color))

# %%

for alpha in [0, 10.0, 50.0]:
    plt.figure(figsize=(4, 3))
    plt.axes([0.1, 0.1, 0.9, 0.9])
    style_figs.light_axis()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.ylabel("y", size=22, weight=600)
    plt.xlabel("x", size=22, weight=600)
    plt.plot(X_all.ravel(), y_all, "o", markersize=15, c=".5", alpha=0.2)
    for X_train, y_train, color in training_sets:
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=color)
        plt.savefig(
            DIR2FIG / f"lasso_alpha_{int(alpha)}.svg",
            facecolor="none",
            edgecolor="none",
        )


# %%
plt.figure(figsize=(4, 3))
plt.axes([0.1, 0.1, 0.9, 0.9])
style_figs.light_axis()
plt.xlim(xlims)
plt.ylim(ylims)
plt.ylabel("y", size=22, weight=600)
plt.xlabel("x", size=22, weight=600)
# plt.plot(X_all.ravel(), y_all, "o", markersize=15, c=".5", alpha=0.2)
X_train, y_train, color = next(iter(training_sets))
plt.plot(X_train.ravel(), y_train, "o", c=color, markersize=15)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=color)
plt.savefig(DIR2FIG / "linreg_noreg_0_nogrey.svg", facecolor="none", edgecolor="none")


# %%
for i, (X_train, y_train, color) in enumerate(training_sets):
    plt.figure(figsize=(4, 3))
    plt.axes([0.1, 0.1, 0.9, 0.9])
    style_figs.light_axis()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.ylabel("y", size=22, weight=600)
    plt.xlabel("x", size=22, weight=600)
    plt.plot(X_all.ravel(), y_all, "o", markersize=15, c=".5", alpha=0.2)
    plt.plot(X_train.ravel(), y_train, "o", c=color, markersize=15)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=color)
    plt.savefig(DIR2FIG / f"linreg_noreg_{i}.svg", facecolor="none", edgecolor="none")

# %%
plt.figure(figsize=(4, 3))
plt.axes([0.1, 0.1, 0.9, 0.9])
style_figs.light_axis()
plt.xlim(xlims)
plt.ylim(ylims)
plt.ylabel("y", size=22, weight=600)
plt.xlabel("x", size=22, weight=600)
plt.plot(X_all.ravel(), y_all, "o", markersize=15, c=".5", alpha=0.2)
X_train, y_train, color = next(iter(training_sets))
plt.plot(X_train.ravel(), y_train, "o", c=color, markersize=15)

regr = linear_model.Lasso(alpha=10)
regr.fit(X_train, y_train)
plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=color)
plt.savefig(DIR2FIG / "lasso_0_withreg.svg", facecolor="none", edgecolor="none")
