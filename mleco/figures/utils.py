import numpy as np

import matplotlib.pyplot as plt

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
