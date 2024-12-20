# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# %%
from sklearn.datasets import fetch_openml

survey = fetch_openml(data_id=534, as_frame=True)
df = survey.frame
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
regressor = Lasso(alpha=0.1)
pipeline = make_pipeline(preprocessor, regressor)
pipeline
# %% [markdown]
# # First
# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipeline.fit(X_train, y_train)

hat_y_test = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, hat_y_test)
print(f"Test mean absolute error: {mae:.2f}")
# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Test score: {score:.2f}")
