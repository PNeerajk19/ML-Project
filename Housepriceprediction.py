#Dragan Real Estate

import pandas as pd
import matplotlib.pyplot as plt
housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()
#%matplotlib inline
housing.hist(bins=50, figsize=(20,15))
#train Test Splitting
import numpy as np

"""
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
"""

#using sci-kit learn
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

"""from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]"""

#Looking for correlations

corr_matrix = housing.corr()

corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]

"""
Scikit-learn Design

Primarily, Three types of objects

1. Estimators- It estimates some parameter based on the dataset eg. imputer
It has a fit method and transform method.
Fit method-  Fits the dataset and calculates internal parameters

2. Transformers- takes input and returns output based on the learning from fit(). 
It alsio has a convenience function called fit_transform() which fits and 
then tranform.

3. Predictors- Linear Regression model is an example of predictor. fit() and  predict()
are two common functions. It also gives score() function which will evaluate the predictions.
"""
"""
Feature Scaling

primarily, two types of feature scaling methods:
1. Min-max scaling(Normalization)
value-min/(max-min)
Sklearn provides a class called MinMaxScaler for this

2. Standardization 
(value-mean)/std
Sklearn provides a class called StandardScaler for this
"""

"""
Creating pipeline
"""
import sklearn
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_num_tr. housing_labels)




