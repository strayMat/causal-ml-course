# %%
from matplotlib import style
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from mleco.constants import DIR2FIG
from mleco.figures import style_figs

# %%
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

fontsize = 15
# Define coordinates for the points
# At t=1
label_treated_t1 = "$E[Y_1(1)|D=1]$"
E_Y11_D1 = (1, 2)
label_control_t1 = "$E[Y_1(0)|D=0]$"
E_Y10_D0 = (1, 4)

# At t=2
label_treated_t2 = "$E[Y_2(1)|D=1]$"
E_Y21_D1 = (2, 1.5)
label_control_t2 = "$E[Y_2(0)|D=0]$"
E_Y20_D0 = (2, 3)
label_treated_counterfactual_t2 = "$E[Y_2(0)|D=1]$"
coef_parallel_trends = E_Y20_D0[1] - E_Y10_D0[1]
E_Y20_D1 = (2, E_Y11_D1[1] + coef_parallel_trends)

# setup without any points
# styling
ax.set_xticks([1, 2])
ax.set_xticklabels(["t = 1", "t = 2"])
# ax.set_yticks([])
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(0.8, 2.35)
ax.set_ylim(0, 5)
# add treatment vline
ax.vlines(x=1.5, ymin=0, ymax=5, linestyles="--", color="grey", alpha=0.5)
label_treatment = ax.text(1.52, 0.05, "Treatment", fontsize=fontsize, color="black")
plt.savefig(DIR2FIG / "did_setup.svg", bbox_inches="tight")
# Plot points
ax.plot(*E_Y10_D0, "ko", label=label_control_t1, color=style_figs.CONTROL_COLOR)
ax.text(
    E_Y10_D0[0],
    E_Y10_D0[1] + 0.15,
    label_control_t1,
    fontsize=fontsize,
    color=style_figs.CONTROL_COLOR,
)
ax.plot(*E_Y11_D1, "ko", label=label_treated_t1, color=style_figs.TREATED_COLOR)
label_treated_t1_factual = "$E[Y_1(1)|D=1]$"
plot_label_treated_t1_factual = ax.text(
    E_Y11_D1[0],
    E_Y11_D1[1] + 0.15,
    label_treated_t1_factual,
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
plt.savefig(DIR2FIG / "did_t1_factual.svg", bbox_inches="tight")
ax.plot(*E_Y20_D0, "ko", label=label_control_t2, color=style_figs.CONTROL_COLOR)
ax.text(
    E_Y20_D0[0],
    E_Y20_D0[1] + 0.15,
    label_control_t2,
    fontsize=fontsize,
    color=style_figs.CONTROL_COLOR,
)
ax.plot(*E_Y21_D1, "ko", label=label_treated_t2, color=style_figs.TREATED_COLOR)
ax.text(
    E_Y21_D1[0],
    E_Y21_D1[1] + 0.15,
    label_treated_t2,
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)

# Draw solid lines for observed paths
ax.plot([1, 2], [E_Y11_D1[1], E_Y21_D1[1]], "k-", color=style_figs.TREATED_COLOR)
ax.plot([1, 2], [E_Y10_D0[1], E_Y20_D0[1]], "k-", color=style_figs.CONTROL_COLOR)
plt.savefig(DIR2FIG / "did_factual.svg", bbox_inches="tight")

# no anticipation assumption
label_treated_t1_no_anticipation = ax.text(
    E_Y11_D1[0],
    E_Y11_D1[1] + 0.15,
    label_treated_t1 + "=$E[Y_1(0)|D=1]$",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
plt.savefig(DIR2FIG / "did_no_anticipation.svg", bbox_inches="tight")

# Draw dashed lines for parallel trends assumption
ax.plot(
    [1, 2],
    [E_Y10_D0[1], E_Y20_D0[1]],
    "b--",
    label="Parallel Trends",
    color="black",
    alpha=0.5,
)
ax.plot(
    [1, 2],
    [E_Y11_D1[1], E_Y20_D1[1]],
    "b--",
    label="Parallel Trends",
    color="black",
    alpha=0.5,
)
ax.plot(*E_Y20_D1, "ko", label=label_treated_t2, color=style_figs.TREATED_COLOR)
ax.text(
    E_Y20_D1[0],
    E_Y20_D1[1] - 0.05,
    label_treated_counterfactual_t2 + "\n counterfactual",
    va="top",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
plt.savefig(DIR2FIG / "did_parallel_trends.svg", bbox_inches="tight")
# plot trend coefficients
trend_control = E_Y20_D0[1] - E_Y10_D0[1]
ax.vlines(
    x=E_Y10_D0[0],
    ymin=E_Y10_D0[1],
    ymax=E_Y10_D0[1] + trend_control,
    color=style_figs.CONTROL_COLOR,
    linestyle="dotted",
)
ax.text(
    E_Y10_D0[0] + 0.01,
    E_Y10_D0[1] + trend_control / 2,
    "Trend(0)",
    fontsize=fontsize,
    color=style_figs.CONTROL_COLOR,
)
trend_treated = E_Y20_D1[1] - E_Y11_D1[1]
counterfactual_trend_plot = ax.vlines(
    x=E_Y11_D1[0],
    ymin=E_Y11_D1[1],
    ymax=E_Y11_D1[1] + trend_treated,
    color=style_figs.TREATED_COLOR,
    linestyle="dotted",
)
counterfactual_trend_text = ax.text(
    E_Y11_D1[0] + 0.01,
    E_Y11_D1[1] + trend_treated / 2,
    "Trend(1)",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
plt.savefig(DIR2FIG / "did_parallel_trends_w_coefs.svg", bbox_inches="tight")

counterfactual_trend_text.remove()
counterfactual_trend_plot.remove()
# no anticipation assumption
plot_label_treated_t1_factual.remove()

label_treated_t1_no_anticipation.remove()
ax.text(
    E_Y11_D1[0],
    E_Y11_D1[1] + 0.15,
    "$E[Y_1(0)|D=1]$",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
# ATT identification
tau_att = E_Y21_D1[1] - E_Y11_D1[1] - trend_control
bracket = FancyArrowPatch(
    (E_Y20_D1[0] + 0.01, E_Y20_D1[1]),
    (E_Y20_D1[0] + 0.01, E_Y20_D1[1] + tau_att),
    arrowstyle="|-|",
    linewidth=2,
    color="black",
)
ax.add_patch(bracket)
ax.annotate(
    r"$\tau_{ATT}$",
    xy=(E_Y20_D1[0] + 0.02, E_Y20_D1[1] + tau_att / 2),
    color="black",
)
factual_trend_treated = E_Y21_D1[1] - E_Y11_D1[1]
factual_trend_plot = ax.vlines(
    x=E_Y11_D1[0],
    ymin=E_Y11_D1[1],
    ymax=E_Y11_D1[1] + factual_trend_treated,
    color=style_figs.TREATED_COLOR,
    linestyle="dotted",
)
factual_trend_text = ax.text(
    E_Y11_D1[0] + 0.01,
    E_Y11_D1[1] + factual_trend_treated - 0.01,
    "Factual Trend(1)",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
plt.savefig(DIR2FIG / "did_att.svg", bbox_inches="tight")
# %%
# # twfe link
fig, ax = plt.subplots(figsize=(10, 6))
fontsize_coeffs = 20
ax.set_xticks([1, 2])
ax.set_xticklabels(["t = 1", "t = 2"])
# ax.set_yticks([])
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(0.8, 2.35)
ax.set_ylim(0, 5)
# add treatment vline
ax.vlines(x=1.5, ymin=0, ymax=5, linestyles="--", color="grey", alpha=0.5)
ax.text(1.52, 0.05, "Treatment", fontsize=fontsize, color="black")
# Plot points
ax.plot(*E_Y10_D0, "ko", label=label_control_t1, color=style_figs.CONTROL_COLOR)
ax.text(
    E_Y10_D0[0],
    E_Y10_D0[1] + 0.15,
    label_control_t1,
    fontsize=fontsize,
    color=style_figs.CONTROL_COLOR,
)
ax.plot(*E_Y11_D1, "ko", label=label_treated_t1, color=style_figs.TREATED_COLOR)
ax.text(
    E_Y11_D1[0],
    E_Y11_D1[1] + 0.15,
    label_treated_t1_factual,
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
ax.plot(*E_Y20_D0, "ko", label=label_control_t2, color=style_figs.CONTROL_COLOR)
ax.text(
    E_Y20_D0[0],
    E_Y20_D0[1] + 0.15,
    label_control_t2,
    fontsize=fontsize,
    color=style_figs.CONTROL_COLOR,
)
ax.plot(*E_Y21_D1, "ko", label=label_treated_t2, color=style_figs.TREATED_COLOR)
ax.text(
    E_Y21_D1[0],
    E_Y21_D1[1] + 0.15,
    label_treated_t2,
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
# Draw solid lines for observed paths
ax.plot([1, 2], [E_Y11_D1[1], E_Y21_D1[1]], "k-", color=style_figs.TREATED_COLOR)
ax.plot([1, 2], [E_Y10_D0[1], E_Y20_D0[1]], "k-", color=style_figs.CONTROL_COLOR)
# Draw dashed lines for parallel trends assumption
ax.plot(
    [1, 2],
    [E_Y10_D0[1], E_Y20_D0[1]],
    "b--",
    label="Parallel Trends",
    color="black",
    alpha=0.5,
)
ax.plot(
    [1, 2],
    [E_Y11_D1[1], E_Y20_D1[1]],
    "b--",
    label="Parallel Trends",
    color="black",
    alpha=0.5,
)
ax.plot(*E_Y20_D1, "ko", label=label_treated_t2, color=style_figs.TREATED_COLOR)
ax.text(
    E_Y20_D1[0],
    E_Y20_D1[1] - 0.05,
    label_treated_counterfactual_t2 + "\n counterfactual",
    va="top",
    fontsize=fontsize,
    color=style_figs.TREATED_COLOR,
)
# control trend
trend_control = E_Y20_D0[1] - E_Y10_D0[1]
ax.vlines(
    x=E_Y20_D0[0],
    ymin=E_Y10_D0[1],
    ymax=E_Y10_D0[1] + trend_control,
    color=style_figs.CONTROL_COLOR,
    linestyle="dotted",
)
ax.text(
    E_Y20_D0[0] + 0.01,
    E_Y10_D0[1] + trend_control / 2,
    r"$\lambda$",
    fontsize=fontsize_coeffs,
    color="black",
)
# control fixed effect
ax.vlines(
    x=E_Y11_D1[0] - 0.02,
    ymin=0,
    ymax=E_Y10_D0[1],
    color=style_figs.CONTROL_COLOR,
    linestyle="dotted",
)
ax.text(
    E_Y11_D1[0] - 0.03,
    E_Y10_D0[1] / 3,
    r"$\alpha$",
    ha="right",
    fontsize=fontsize_coeffs,
    color="black",
)
# treated fixed effect
ax.vlines(
    x=E_Y11_D1[0] - 0.05,
    ymin=E_Y11_D1[1],
    ymax=E_Y10_D0[1],
    color=style_figs.TREATED_COLOR,
    linestyle="dotted",
)
ax.text(
    E_Y11_D1[0] - 0.06,
    (E_Y10_D0[1] + E_Y11_D1[1]) / 2,
    r"$\gamma$",
    ha="right",
    fontsize=fontsize_coeffs,
    color="black",
)
# att
ax.vlines(
    x=E_Y20_D1[0],
    ymin=E_Y20_D1[1],
    ymax=E_Y20_D1[1] + tau_att,
    color=style_figs.TREATED_COLOR,
    linestyle="dotted",
)
ax.text(
    E_Y20_D1[0] + 0.01,
    E_Y20_D1[1] + tau_att / 2,
    r"$\tau_{ATT}$",
    fontsize=fontsize_coeffs,
    color="black",
)
# labels for coefficients
ax.text(
    E_Y21_D1[0],
    4.8,
    "Time effects",
    ha="center",
    fontsize=fontsize,
    color="black",
)
ax.text(
    E_Y11_D1[0],
    4.8,
    "Fixed effects",
    ha="center",
    fontsize=fontsize,
    color="black",
)
plt.savefig(DIR2FIG / "did_twfe.svg", bbox_inches="tight")
# %%
# Non parallel trends
fig, ax = plt.subplots(figsize=(12, 5))
fontsize_coeffs = 20
x = [1, 2, 3, 4]
ax.set_xticks(x)
ax.set_xticklabels([f"t = {x_}" for x_ in x])

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(0.8, 4.35)
# ax.set_ylim(0, 5)
# add treatment vline
ax.vlines(x=3.5, ymin=0, ymax=200, linestyles="--", color="grey", alpha=0.5)
ax.text(3.52, 0.05, "Treatment", fontsize=fontsize, color="black")
# Plot points

E_Y10_D0 = 170
E_Y20_D0 = 200

E_Y11_D1 = 45
E_Y21_D1 = 40
plt.plot(x[2:], [E_Y10_D0, E_Y20_D0], color=style_figs.CONTROL_COLOR, lw=2)
plt.plot(x[2:], [E_Y11_D1, E_Y21_D1], color=style_figs.TREATED_COLOR, lw=2)
# counterfactual with parallele trends
plt.plot(
    [3, 4],
    [E_Y11_D1, E_Y11_D1 + (E_Y20_D0 - E_Y10_D0)],
    label="Counterfactual",
    lw=2,
    color="black",
    ls="--",
)
ax.grid()
plt.savefig(DIR2FIG / "did_non_parallel_trends_last_periods.svg", bbox_inches="tight")
plt.plot(x[:3], [120, 150, E_Y10_D0], color=style_figs.CONTROL_COLOR, lw=2)
plt.plot(x[:3], [60, 50, E_Y11_D1], color=style_figs.TREATED_COLOR, lw=2)
plt.savefig(DIR2FIG / "did_non_parallel_trends_all_periods.svg", bbox_inches="tight")
# %%