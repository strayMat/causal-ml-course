"""
Goal : show that two different causal structures can yield identical observational data. We cannot distinguish them without interventions.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mleco.constants import DIR2FIG

# Set seed for reproducibility
np.random.seed(42)
N = 1000  # Number of employees

# ==========================================
# World A: The Chain (E -> S -> I)
# Education causes Skill, Skill causes Income
# ==========================================
# Structural Equations:
# E = N(0, 1)
# S = 0.8 * E + N(0, 0.6)  (S depends on E)
# I = 0.8 * S + N(0, 0.6)  (I depends on S)

U_e_A = np.random.normal(0, 1, N)
U_s_A = np.random.normal(0, 0.6, N)
U_i_A = np.random.normal(0, 0.6, N)

E_A = U_e_A
S_A = 0.8 * E_A + U_s_A
I_A = 0.8 * S_A + U_i_A

df_A = pd.DataFrame({"Education": E_A, "Skill": S_A, "Salary": I_A})

# ==========================================
# World B: The Fork (E <- S -> I)
# Skill causes *both* Education and Income.
# ==========================================
# Structural Equations:
# S = N(0, 1) (Skill is the root cause here)
# E = 0.8 * S + N(0, 0.6)  (E depends on S)
# I = 0.8 * S + N(0, 0.6)  (I depends on S)

U_s_B = np.random.normal(0, 1, N)  # Skill is fundamental here
U_e_B = np.random.normal(0, 0.6, N)
U_i_B = np.random.normal(0, 0.6, N)

S_B = U_s_B
E_B = 0.8 * S_B + U_e_B
I_B = 0.8 * S_B + U_i_B

df_B = pd.DataFrame({"Education": E_B, "Skill": S_B, "Salary": I_B})


# ==========================================
# The Comparison: Observational Data
# ==========================================
print("--- Observational Correlation Matrix (World A: E -> S -> I) ---")
cor_A = df_A.corr()
print(cor_A.round(3))
print("\n" + "=" * 50 + "\n")
print("--- Observational Correlation Matrix (World B: E <- S -> I) ---")
cor_B = df_B.corr()
print(cor_B.round(3))

# Let's look at the difference between the correlation matrices
print("\n" + "=" * 50 + "\n")
print("--- Difference in Correlations (World A - World B) ---")
print("The differences are practically zero noise:")
print((cor_A - cor_B).round(4))


# %%
from mleco.figures.style_figs import *

# Optional Visual check - Pairplot with correlations
# Create combined dataframe with world labels
df_A_labeled = df_A.copy()
df_A_labeled["DGP"] = "1: Chain"

df_B_labeled = df_B.copy()
df_B_labeled["DGP"] = "2: Fork"

df_combined = pd.concat([df_A_labeled, df_B_labeled])

# Create pairplot using seaborn
g = sns.pairplot(
    data=df_combined,
    hue="DGP",
    diag_kind="hist",
    plot_kws={"alpha": 0.6, "s": 10},
    diag_kws={"alpha": 0.4},
    corner=True,
)
handles, labels = g.axes[1, 0].get_legend_handles_labels()
g.legend.remove()
g.figure.legend(
    handles=handles,
    labels=labels,
    markerscale=5,
    fontsize="large",
    title="DGP",
    loc="upper right",
    bbox_to_anchor=(
        0.8,
        0.8,
    ),  # Adjust these coordinates to center it in the gap
)
# handles, labels = g.axes[0, 0].get_legend_handles_labels()
# g.legend.remove()
# plt.legend(
#     handles=handles,
#     labels=labels,
#     title="DGP",
#     loc="upper right",
# )
plt.savefig(
    DIR2FIG / "5_dgp_equivalence.svg",
    bbox_inches="tight",
)
# plt.show() # Uncomment to view plots
# %%

# ==========================================
# THE LESSON: The Effect of Intervention
# ==========================================
print("\n" + "=" * 50 + "\n")
print("THE LESSON: Intervening on Education (E)")
print("We force everyone's Education score up by 2 standard deviations.")
print("We then see what happens to Salary (I) in both worlds.")

# Intervention in World A (E -> S -> I)
# If we set E, S changes, and therefore I changes.
E_intervened = E_A + 2.0
S_post_intervention_A = 0.8 * E_intervened + U_s_A  # S uses new E
I_post_intervention_A = 0.8 * S_post_intervention_A + U_i_A  # I uses new S
lift_A = I_post_intervention_A.mean() - I_A.mean()

# Intervention in World B (E <- S -> I)
# If we set E, we break the arrow S -> E.
# S does not change. Therefore I does not change.
E_intervened_B = E_B + 2.0
# S remains whatever it was originally
S_post_intervention_B = S_B
# I remains determined by S, regardless of what we did to E
I_post_intervention_B = 0.8 * S_post_intervention_B + U_i_B
lift_B = I_post_intervention_B.mean() - I_B.mean()

print(f"\nExpected Salary Lift in World A (Chain): {lift_A:.4f}")
print(f"Expected Salary Lift in World B (Fork):  {lift_B:.4f}")
