# %%
from docutils.nodes import target
import pandas as pd

# %%
# We fetch the data from `OpenML <http://openml.org/>`_.
# Note that setting the parameter `as_frame` to True will retrieve the data
# as a pandas dataframe.
from sklearn.datasets import fetch_openml

adult_census_data = fetch_openml(data_id=1590, as_frame=True)
# %% [markdown]
# Inspection of the dataset

# ```{note}
# We use the [Pandas](https://pandas.pydata.org/) Python library to work
# manipulate 1 and 2 dimensional structured data. If you have never used
# pandas, we recommend you look at this
# [tutorial](https://pandas.pydata.org/docs/user_guide/10min.html).
# ```
# %%
# The adult_census_data object is a dictionary-like object which contains:
print(adult_census_data.keys())
# %%
# We can read a description of the dataset using the DESCR key of the
print(adult_census_data.DESCR)
# %%
# Let's look at the dataframe itself
adult_census = adult_census_data.frame
# Print the dimensions and the first few rows of the dataframe
print(adult_census.shape)
adult_census.head()
# %% [markdown]
## Focus on the target variable
# %%
# The target is the class variable: 'class'
target_name = "class"
y = adult_census[target_name]
adult_census[target_name].value_counts()

# %% [markdown]
# ```{note}
# Here, classes are slightly imbalanced, meaning there are more samples of one
# or more classes compared to others. In this case, we have many more samples
# with `" <=50K"` than with `" >50K"`. Class imbalance happens often in practice
# and may need special techniques when building a predictive model.
#
# For example in a medical setting, if we are trying to predict whether subjects
# may develop a rare disease, there would be a lot more healthy subjects than
# ill subjects in the dataset.
# ```
## Visual inspection of the data

### Distinguish numerical and categorical columns
# The dataset contains both numerical and categorical data. Numerical values take continuous values, for example "age". Categorical values can have a finite number of values, for example "native-country".
# %%
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
all_columns = numerical_columns + categorical_columns + [target_name]
adult_census = adult_census[all_columns]

# %% [markdown]
### Display histograms for numerical columns
# Let’s look at the distribution of individual features, to get some insights about the data. We can start by plotting histograms, note that this only works for features containing numerical values:
# %%
_ = adult_census.hist(figsize=(20, 14))
# %% [markdown]
### Display the unique values of the categorical columns

# %%
adult_census["sex"].value_counts()
# %% [mardown]
# Quite unbalanced!
# %%
adult_census["education"].value_counts()
# %% [markdown]
# s noted above, "education-num" distribution has two clear peaks around 10 and 13. It would be reasonable to expect that "education-num" is the number of years of education.

# Let’s look at the relationship between "education" and "education-num".
# %%
pd.crosstab(index=adult_census["education"], columns=adult_census["education-num"])
# %% [markdown]
# Build the predictive model
## Separate the target variable from the data
# %%
# %%
data = adult_census.drop(columns=[target_name])
data
# %% [markdown]
## Select only some numerical columns
# For simplicity, we will select only the numerical columns in this dataset.
data_numeric = data[numerical_columns]
data_numeric
# %% [markdown]
# ## Separate training and testing data
# %%
from sklearn.model_selection import train_test_split

data_numeric_train, data_numeric_test, y_train, y_test = train_test_split(
    data_numeric, y, random_state=42
)
# ## First model
# %%
from sklearn.linear_model import LogisticRegression

# model declaration
model = LogisticRegression()
model
# %%
# model fitting/training
_ = model.fit(data_numeric_train, y_train)
# %% [markdown]
# ## Remarks on fitting a model

# # ![Predictor fit diagram](../slides/img/ML_1/api_diagram-predictor.fit.svg)
#  In scikit-learn an object that has a fit method is called an estimator. The method fit is composed of two elements: (i) a learning algorithm and (ii) some model states. The learning algorithm takes the training data and training target as input and sets the model states. These model states are later used to either predict (for classifiers and regressors) or transform data (for transformers).
# %%
# model prediction
y_train_predicted = model.predict(data_numeric_train)
# %% [markdown]
# # ![Predictor predict diagram](../slides/img/ML_1/api_diagram-predictor.predict.svg)

# %%
# model evaluation (by hand):
print(
    "Number of correct prediction: "
    f"{(y_train[:5] == y_train_predicted[:5]).sum()} / 5"
)
# %% [markdown]
# To get a better assessment, we can compute the average success rate.
# %%
(y_train == y_train_predicted).mean()

# %% [markdown]
# ## Evaluation of the model
# %%
accuracy = model.score(data_numeric_test, y_test)
model_name = model.__class__.__name__
print(f"The test accuracy using a {model_name} is {accuracy:.3f}")

# %% [markdown]
# We use the generic term model for objects whose goodness of fit can be measured using the score method. Let’s check the underlying mechanism when calling score:
# # ![Predictor score diagram](../slides/img/ML_1/api_diagram-predictor.score.svg)
# %%
