# %% [markdown]
# Figures for introducing trees.
# %%
# tree splitting choice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

import mleco.figures.style_figs
from mleco.figures.utils import (
    fit_and_plot_decision_tree,
    plot_impurities,
    get_impurities_from_tree,
    gini_criteria,
)
from mleco.constants import DIR2FIG

# %%
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0
)
data = pd.DataFrame(
    np.hstack((X, y.reshape(-1, 1))), columns=["feature1", "feature2", "target"]
)
# random split
model = DecisionTreeClassifier(max_depth=1, splitter="random", random_state=0)
model.fit(data[["feature1", "feature2"]], data["target"])
fig, ax = plt.subplots()
fit_and_plot_decision_tree(
    data,
    model,
    ["feature1", "feature2"],
    "target",
    ax=ax,
)
l_impurity, r_impurity, f_impurity = get_impurities_from_tree(model)
plot_impurities(l_impurity, r_impurity, f_impurity, ax=ax)
plt.savefig(DIR2FIG / "tree_random_split.svg", bbox_inches="tight")
# %%
# # next split hack to change the threshold
for i in range(1, 20):
    # get the threshold from the model, find the next threshold
    f_split_feature = model.tree_.feature[0]
    f_split_threshold = model.tree_.threshold[0]
    right_index = X[:, f_split_feature] > f_split_threshold
    r_samples = X[right_index, :]
    n_threshold = np.sort(r_samples[:, f_split_feature])[1]
    # Compute the impurities at the current split
    r_labels = y[right_index]
    l_labels = y[~right_index]
    l_impurity = gini_criteria(l_labels)
    r_impurity = gini_criteria(r_labels)
    f_impurity = (
        len(l_labels) / len(y) * l_impurity + len(r_labels) / len(y) * r_impurity
    )
    # set the next threshold
    model.tree_.threshold[0] = n_threshold
    if i in [2, 10, 19]:
        fig, ax = plt.subplots()
        fit_and_plot_decision_tree(
            data,
            model,
            ["feature1", "feature2"],
            "target",
            fit=False,
            ax=ax,
        )
        plot_impurities(l_impurity, r_impurity, f_impurity, ax=ax)
        plt.savefig(DIR2FIG / f"tree_split_{i}.svg", bbox_inches="tight")

# %%
# best split
model = DecisionTreeClassifier(max_depth=1, splitter="best", random_state=0)
fig, ax = plt.subplots()
fit_and_plot_decision_tree(
    data,
    model,
    ["feature1", "feature2"],
    "target",
    fit=True,
    ax=ax,
)
l_impurity, r_impurity, f_impurity = get_impurities_from_tree(model)
plot_impurities(l_impurity, r_impurity, f_impurity, ax=ax)
plt.savefig(DIR2FIG / "tree_best_split.svg", bbox_inches="tight")
# %%
