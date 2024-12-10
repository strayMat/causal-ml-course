# %%
"""
Simple example of overfit with splines
"""

from docutils.languages import fr
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from mleco.constants import DIR2FIG
from mleco.figures import style_figs

from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-200:3]
diabetes_X_test = diabetes_X_temp[-200:].T

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-200:3]
diabetes_y_test = diabetes.target[-200:]

# Sort the data and remove duplicates (for interpolation)
order = np.argsort(diabetes_X_train.ravel())
X_train = diabetes_X_train.ravel()[order]
y_train = diabetes_y_train[order]
# Avoid duplicates
y_train_ = list()
for this_x in np.unique(X_train):
    y_train_.append(np.mean(y_train[X_train == this_x]))
X_train = np.unique(X_train)

y_train = np.array(y_train_)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train.reshape((-1, 1)), y_train)

plt.figure(1, figsize=(0.8 * 4, 0.8 * 3), facecolor="none")
# Plot with test data
plt.clf()
ax = plt.axes([0.1, 0.1, 0.9, 0.9])

plt.scatter(X_train, y_train, color="k", s=9)

plt.plot(
    [-0.08, 0.12],
    regr.predict(
        [
            [
                -0.08,
            ],
            [
                0.12,
            ],
        ]
    ),
    linewidth=3,
)
# Add the R-squared scores to the plot
r2_train = r2_score(diabetes_y_train, regr.predict(diabetes_X_train.reshape(-1, 1)))
r2_test = r2_score(diabetes_y_test, regr.predict(diabetes_X_test.reshape(-1, 1)))
plt.text(
    0.05,
    0.95,
    f"Train R²: {r2_train:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.axis("tight")
ymin, ymax = plt.ylim()
style_figs.light_axis()
plt.ylabel("y", size=16, weight=600)
plt.xlabel("x", size=16, weight=600)

plt.savefig(DIR2FIG / "ols_simple_w_r2.svg", facecolor="none", edgecolor="none")

plt.text(
    0.05,
    0.80,
    f"Test R²: {r2_test:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(facecolor="C1", alpha=0.5),
)
plt.scatter(diabetes_X_test, diabetes_y_test, color="C1", s=9)
plt.ylim(ymin, ymax)
plt.xlim(-0.08, 0.12)

plt.savefig(DIR2FIG / "ols_test_w_r2.svg", facecolor="none", edgecolor="none")


# Plot cubic splines
plt.clf()
ax = plt.axes([0.1, 0.1, 0.9, 0.9])
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    SplineTransformer(degree=5, n_knots=25, knots="quantile"), LinearRegression()
)
model.fit(diabetes_X_train, diabetes_y_train)

plt.scatter(X_train, y_train, color="k", s=9, zorder=20)
x_spline = np.linspace(-0.08, 0.12, 600)

y_spline = model.predict(x_spline.reshape(-1, 1))
# Add the R-squared scores to the plot
r2_train = r2_score(diabetes_y_train, model.predict(diabetes_X_train.reshape(-1, 1)))
r2_test = r2_score(diabetes_y_test, model.predict(diabetes_X_test.reshape(-1, 1)))
plt.text(
    0.05,
    0.95,
    f"Train R²: {r2_train:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.axis("tight")
plt.xlim(-0.08, 0.12)
plt.ylim(ymin, ymax)

style_figs.light_axis()

plt.ylabel("y", size=16, weight=600)
plt.xlabel("x", size=16, weight=600)


plt.savefig(DIR2FIG / "splines_cubic_w_r2.svg", facecolor="none", edgecolor="none")

plt.text(
    0.05,
    0.80,
    f"Test R²: {r2_test:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(facecolor="C1", alpha=0.5),
)
plt.plot(x_spline, y_spline, linewidth=3)

plt.scatter(diabetes_X_test, diabetes_y_test, color="C1", s=9)
plt.savefig(DIR2FIG / "splines_test_w_r2.svg", facecolor="none", edgecolor="none")

plt.show()
