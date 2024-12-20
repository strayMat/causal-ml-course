# %%
from IPython.extensions import autoreload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mleco.figures.utils import plot_train_test_indices
from mleco.constants import DIR2FIG
%load_ext autoreload
%autoreload 2
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
        ax2.scatter([0]*i, saved_mae[:i], c='blue', alpha=0.8, s=score_point_size, marker="_")
    mae_scatter = ax2.scatter([0], [mae], c='black', alpha=0.8, s=score_point_size, marker="_")
    mae_label = ax2.annotate(f"{mae:.2f}", (0, mae), textcoords="offset points", xytext=(10,0), ha='left')
    # formattting
    ax1.set_yticks([-rs for rs in random_states])
    ax1.set_yticklabels(random_states)
    ax1.set_ylim(-len(random_states), 1)

    # Add labels and legend
    ax1.set_ylabel('Random seed')

    ax2.set_ylim(min(saved_mae)-0.1, max(saved_mae)+0.1)
    ax2.set_xticks([])
    ax2.yaxis.tick_right()
    ax2.grid(False)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Mean Absolute Error')
    plt.savefig(
        DIR2FIG/f'train_test_split_visualization_seed_{rs}.png',
        bbox_inches="tight"
        )
    mae_label.remove()
    mae_scatter.remove()
# %%w