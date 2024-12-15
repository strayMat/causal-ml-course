# %%
import numpy as np
from scipy.datasets import face
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
for i, edge_color in zip(range(6), colors):
    X_train, _, y_train, _ = train_test_split(
        X_all, y_all, train_size=train_size, random_state=rng
    )
    training_sets.append((X_train, y_train, edge_color))

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
    for X_train, y_train, edge_color in training_sets:
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=edge_color)
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
X_train, y_train, edge_color = next(iter(training_sets))
plt.plot(X_train.ravel(), y_train, "o", c=edge_color, markersize=15)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=edge_color)
plt.savefig(DIR2FIG / "linreg_noreg_0_nogrey.svg", facecolor="none", edgecolor="none")


# %%
for i, (X_train, y_train, edge_color) in enumerate(training_sets):
    plt.figure(figsize=(4, 3))
    plt.axes([0.1, 0.1, 0.9, 0.9])
    style_figs.light_axis()
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.ylabel("y", size=22, weight=600)
    plt.xlabel("x", size=22, weight=600)
    plt.plot(X_all.ravel(), y_all, "o", markersize=15, c=".5", alpha=0.2)
    plt.plot(X_train.ravel(), y_train, "o", c=edge_color, markersize=15)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=edge_color)
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
X_train, y_train, edge_color = next(iter(training_sets))
plt.plot(X_train.ravel(), y_train, "o", c=edge_color, markersize=15)

regr = linear_model.Lasso(alpha=10)
regr.fit(X_train, y_train)
plt.plot(X_test, regr.predict(X_test), linewidth=1.5, c=edge_color)
plt.savefig(DIR2FIG / "lasso_0_withreg.svg", facecolor="none", edgecolor="none")

# %%
# lasso intuition
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as mcolors


# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
# Set up the axes limits and aspect ratio
ax.set_xlim(-150, 300)
ax.set_ylim(-130, 300)
ax.set_aspect("equal")

# Draw axes
ax.plot([-300, 350], [0, 0], color="black", linewidth=2)  # Horizontal axis
ax.plot([0, 0], [-250, 300], color="black", linewidth=2)  # Vertical axis

# Add axis labels
ax.text(290, 10, r"$\beta_1$", fontsize=16, color="black")
ax.text(10, 230, r"$\beta_2$", fontsize=16, color="black")


# Function to draw ellipses with a rotation and center shift
def draw_ellipse(
    ax, center, width, height, angle, color, facecolor=None, linestyle="-"
):
    ellipse = patches.Ellipse(
        center,
        width,
        height,
        angle=angle,
        edgecolor=color,
        fill=True,
        facecolor=facecolor,
        linestyle=linestyle,
        linewidth=2,
        alpha=0.5,
    )
    ax.add_patch(ellipse)


# Rotated ellipses (rotated 90 degrees, centered at (0, 0))
center = (120, 160)

angle = 30
ax.text(
    center[0],
    center[0] + 70,
    r"Min of MSE: $\hat{\beta}_{OLS}$",
    fontsize=12,
    color="black",
    va="top",
    ha="center",
)
edge_color = "black"
# Origin marker
ax.plot(0, 0, "o", color="black")  # Origin marker
ax.text(-40, 10, "(0,0)", fontsize=12, color="black", ha="left")
# Hide axes ticks
ax.axis("off")

## draw a cross at the center
ax.plot(center[0], center[1], "x", color="black")
draw_ellipse(
    ax, center, 180, 120, angle, edge_color, facecolor=mcolors.CSS4_COLORS["lightcoral"]
)  # Inner ellipse (dotted)
plt.savefig(DIR2FIG / "lasso_intuition_inner.svg", facecolor="none", edgecolor="none")

draw_ellipse(
    ax,
    center,
    240,
    160,
    angle,
    edge_color,
    facecolor=mcolors.CSS4_COLORS["lightsalmon"],
)  # Middle ellipse (dashed)
plt.savefig(DIR2FIG / "lasso_intuition_middle.svg", facecolor="none", edgecolor="none")

draw_ellipse(
    ax,
    center,
    300,
    200,
    angle,
    edge_color,
    facecolor=mcolors.CSS4_COLORS["lemonchiffon"],
)  # Outer ellipse
plt.savefig(DIR2FIG / "lasso_intuition_outer.svg", facecolor="none", edgecolor="none")

penalty_color = mcolors.CSS4_COLORS["lightgreen"]
# Lasso penalty diamond
diamond_diagonal = 72

penalty_color = mcolors.CSS4_COLORS["forestgreen"]
# Lasso penalty diamond
diamond_points = np.array(
    [
        [0, -diamond_diagonal],
        [diamond_diagonal, 0],
        [0, diamond_diagonal],
        [-diamond_diagonal, 0],
    ]
)
diamond = patches.Polygon(
    diamond_points,
    closed=True,
    edgecolor=penalty_color,
    facecolor=penalty_color,
    linewidth=2,
    alpha=0.5,
)
diamond = ax.add_patch(diamond)

label_lasso = ax.text(
    50, -50, r"Lasso penalty: $|\beta_0|+|\beta_1| \leq t$", fontsize=12, color="black"
)
plt.savefig(DIR2FIG / "lasso_intuition_penalty.svg", facecolor="none", edgecolor="none")
# Remove the diamond
diamond.remove()
label_lasso.remove()
# draw an circle of rayon 75 at 0, 0 the axes
circle = patches.Circle(
    (0, 0),
    60,
    color=penalty_color,
    fill=True,
    linestyle="-",
    linewidth=2,
    alpha=0.5,
)
ax.add_patch(circle)
ax.text(
    50, -50, r"Ridge penalty: $\beta_0^2+\beta_1^2 \leq t$", fontsize=12, color="black"
)
plt.savefig(DIR2FIG / "ridge_intuition_penalty.svg", facecolor="none", edgecolor="none")
# %%
