import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay


cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm


def plot_train_test_indices(X_train_indices, X_test_indices, ax, y=0, lw=10):
    indices = np.array([np.nan] * (len(X_train_indices) + len(X_test_indices)))
    indices[X_train_indices] = 1
    indices[X_test_indices] = 0

    # Visualize the results
    ax.scatter(
        range(len(indices)),
        [y] * len(indices),
        c=indices,
        marker="_",
        lw=lw,
        cmap=cmap_cv,
        vmin=-0.2,
        vmax=1.2,
    )
    # Formatting
    ax.set(
        xlabel="Sample index",
        # yticks=[y],
        xlim=[0, len(indices)],
    )
    # ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    use_groups = "Group" in type(cv).__name__
    groups = group if use_groups else None
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


# utils for fitting and plotting a tree
def fit_and_plot_decision_tree(
    data, model, feature_names, target_name, ax=None, fit=True, plot_impurities=False
):
    """
    Compute decision boundary for the first node of a binary classification tree.
     - If asked, fit the model to the data.
     - If asked plot the impurities (left, right and full).
    """
    X = data[feature_names]
    y = data[target_name]
    if fit:
        model.fit(X, y)
    palette = ["tab:red", "tab:blue"]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict",
        cmap="RdBu",
        alpha=0.5,
        ax=ax,
    )
    sns.scatterplot(
        ax=ax,
        data=data,
        x=feature_names[0],
        y=feature_names[1],
        hue=target_name,
        palette=palette,
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax


def gini_criteria(y):
    """
    Compute the gini impurity for a binary classification problem.
    """
    n = len(y)
    if n == 0:
        return 0
    p = np.mean(y)
    return 2 * p * (1 - p)


def plot_impurities(l_impurity, r_impurity, f_impurity, ax):
    """Plot the impurities on the top of the plot.
    Args:
        l_impurity (_type_): _description_
        r_impurity (_type_): _description_
        f_impurity (_type_): _description_
        ax (_type_): _description_
    """
    font_size = 14
    ax.text(
        0.01,
        1.01,
        f"Left impurity: {l_impurity:.2f}",
        transform=ax.transAxes,
        ha="left",
        fontsize=font_size,
        color="red",
    )
    ax.text(
        1,
        1.01,
        f"Right impurity: {r_impurity:.2f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=font_size,
        color="blue",
    )
    ax.text(
        0.5,
        1.1,
        f"Full impurity: {f_impurity:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=font_size,
        color="black",
    )
    return ax


def get_impurities_from_tree(model):
    """
    Compute the value of the criterion for the first two nodes (left and right) of the fitted tree.
    """
    left_node_index = model.tree_.children_left[0]
    right_node_index = model.tree_.children_right[0]

    left_node_criterion = model.tree_.impurity[left_node_index]
    right_node_criterion = model.tree_.impurity[right_node_index]

    left_n_node_samples = model.tree_.n_node_samples[left_node_index]
    right_n_node_samples = model.tree_.n_node_samples[right_node_index]

    impurity = (
        left_node_criterion * left_n_node_samples / model.tree_.n_node_samples[0]
        + right_node_criterion * right_n_node_samples / model.tree_.n_node_samples[0]
    )
    return left_node_criterion, right_node_criterion, impurity
