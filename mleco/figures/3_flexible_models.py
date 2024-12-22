# %%
from IPython.extensions import autoreload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mleco.figures.utils import plot_train_test_indices
from mleco.constants import DIR2FIG

# %%
from sklearn.datasets import fetch_openml

survey = fetch_openml(data_id=534, as_frame=True)
df = survey.frame
print(df.shape)
df.head()
# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
categorical_columns = [
    "SOUTH",
    "SEX",
    "UNION",
    "RACE",
    "OCCUPATION",
    "SECTOR",
    "MARR",
]
numerical_preprocessor = StandardScaler()
numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns),
    ]
)
preprocessor.fit(df)
transformed_df = pd.DataFrame(
    preprocessor.transform(df), columns=preprocessor.get_feature_names_out()
)
print(transformed_df.shape)
transformed_df.head()
# %%
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline

X = df.drop(columns="WAGE")
y = df["WAGE"]
regressor = Lasso(alpha=10)
pipeline = make_pipeline(preprocessor, regressor)
pipeline
# %% [markdown]
# # First
# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

saved_mae = []
saved_x_train_indices = []
saved_x_test_indices = []
random_states = range(10)

for rs in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)
    pipeline.fit(X_train, y_train)
    hat_y_test = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, hat_y_test)
    saved_mae.append(mae)
    saved_x_train_indices.append(X_train.index)
    saved_x_test_indices.append(X_test.index)
# %%
# plot figure
fig = plt.figure(figsize=(6, 3))
gs = fig.add_gridspec(1, 4)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3])
score_point_size = 100

for i, rs in enumerate(random_states):
    X_train_indices = saved_x_train_indices[i]
    X_test_indices = saved_x_test_indices[i]
    mae = saved_mae[i]
    ax1 = plot_train_test_indices(X_train_indices, X_test_indices, ax1, y=-rs, lw=10)
    if i >= 1:
        ax2.scatter(
            [0] * i, saved_mae[:i], c="blue", alpha=0.8, s=score_point_size, marker="_"
        )
    mae_scatter = ax2.scatter(
        [0], [mae], c="black", alpha=0.8, s=score_point_size, marker="_"
    )
    mae_label = ax2.annotate(
        f"{mae:.2f}", (0, mae), textcoords="offset points", xytext=(10, 0), ha="left"
    )
    # formattting
    ax1.set_yticks([-rs for rs in random_states])
    ax1.set_yticklabels(random_states)
    ax1.set_ylim(-len(random_states), 1)

    # Add labels and legend
    ax1.set_ylabel("Random seed")

    ax2.set_ylim(min(saved_mae) - 0.1, max(saved_mae) + 0.1)
    ax2.set_xticks([])
    ax2.yaxis.tick_right()
    ax2.grid(False)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Mean Absolute Error")
    plt.savefig(
        DIR2FIG / f"train_test_split_visualization_seed_{rs}.png", bbox_inches="tight"
    )
    mae_label.remove()
    mae_scatter.remove()
print(np.mean(saved_mae), np.std(saved_mae))
# %%w
# ## GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_distributions = {
    "lasso__alpha": loguniform(1e-6, 1e3),
}

model_random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    verbose=1,
    scoring="neg_mean_absolute_error",
)
model_random_search.fit(X, y)
# %%

model_random_search.cv_results_.keys()
# %%
# Plotting the CV scores
cv_results = model_random_search.cv_results_
alphas = cv_results["param_lasso__alpha"]

# We used neg median absolute error as the scoring metric, so we reverse it.
mean_test_scores = -cv_results["mean_test_score"]
std_test_scores = cv_results["std_test_score"]

plt.figure(figsize=(6, 3))
plt.axvspan(1e-5 / 2, 1e-1, color="lightgreen", alpha=0.4, label="sweet spot")
sc = plt.scatter(alphas, mean_test_scores, edgecolor="k", s=100)
plt.errorbar(
    alphas,
    mean_test_scores,
    yerr=std_test_scores,
    fmt="o",
    color="black",
    alpha=0.5,
    capsize=5,
)
plt.legend(prop={"size": 16}, loc="lower right")
plt.xscale("log")
font_size = 18
plt.xlabel("Alpha (log scale)", fontsize=font_size)
plt.ylabel("Median Absolute\nError on test", fontsize=font_size)

plt.grid(True)
plt.minorticks_off()
plt.savefig(DIR2FIG / "lasso_random_search_cv.svg", bbox_inches="tight")
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
