# Data Science Project for the first Kaggle Dataset: titanic (https://www.kaggle.com/c/titanic/notebooks)
# Written based on the https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

# Initial imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
import sklearn
import xgboost as xgb
# import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
# import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

# import SklearnHelper.py as skh

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

# Read the data
train = pd.read_csv('./titanic_train.csv', index_col=0)
test = pd.read_csv('./titanic_test.csv', index_col=0)

# Stores the ID of Passengers for the final test
PassengerId = test.index

# Show all the columns
pd.set_option('display.max_columns', None)

# Feature engineering part
full_data = [train, test]

# Computes the length of the name of a person
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# True, if the person has a cabin
train['Has_cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

# Fix the missing Age by putting the avg age of class
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[:, 'Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset.loc[:, 'Age'] = dataset['Age'].astype(int)

# Create a New Feature CategoricalAge
train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Fix the missing Embark by choosing the most common Embark
# Fill the missing Embarked with the most common value
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Create new Feature Family Size as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create a feature: is alone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Fills the Empty fields in Fare column with median and creates a feature CategoricalFare by cutting the Fare in 4 Percentile ranges
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""


# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single group "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train['Fare'].describe())
print(train.dtypes)

# Mapping part (get dummy data)
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Titles
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Fare (try using cut in the future?)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.545), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.545) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature selection
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Name_length']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)

# Test: Print the resulting dataframe
# print(train.head(3))

colormap = plt.cm.RdBu
plt.figure('Pearson Correlation of Features', figsize=(14, 12))
plt.title('Pearson Correlation of Features')
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5  # sets folds for out of fold prediction
# kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)
kf = KFold(NFOLDS) # , random_state=SEED)
kf.get_n_splits(ntrain)


# kf.split(kf)
# print(enumerate(kf))

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train_fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


# Cross-validation (K-fold cross valid.)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    # original version: for i, (train_index, test_index) in enumerate(kf):

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        print(train_index, test_index)
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train_fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Setting the parameters for the 1st line models
# Random Forest parameters

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    # 'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra trees parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    # 'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Ada Boost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
    # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

# Creating the 5 Objects that represent our 4 Models
rf = SklearnHelper(clf=RandomForestClassifier, params=rf_params)
# rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, params=gb_params)
svc = SklearnHelper(clf=SVC, params=svc_params)

# Creating the Numpy arrays of the train, test and target dataframes as an input to the models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
X_train = train.values  # Creates an array of the train data
X_test = test.values  # Creates an array of the test data

# Running the first level models, create the OOF train and test predictions. These will be used as new features
et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test)  # Random forest
ada_oof_train, ada_oof_test = get_oof(ada, X_train, y_train, X_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, X_train, y_train, X_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, X_train, y_train, X_test)  # Suport Vector Classifier

print("Training is complete")

# Print the importances

rf_features = rf.feature_importances(X_train, y_train)
et_features = et.feature_importances(X_train, y_train)
ada_features = ada.feature_importances(X_train, y_train)
gb_features = gb.feature_importances(X_train, y_train)

print(rf_features)  # , et_feature, ada_feature, gb_feature)

# Create a dataframe from the importance features for easy plotting

cols = train.columns.values

feature_dataframe = pd.DataFrame({
    'features': cols,
    'Random Forest feature importances': rf_features,
    'Extra Trees feature importances': et_features,
    'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
})

print(feature_dataframe)

# Plotly visualisations of feature importances
# Scatter plot: Random Forest feature importances
trace = go.Scatter(
    x=feature_dataframe['features'].values,
    y=feature_dataframe['Random Forest feature importances'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Random Forest Feature Importance',
    hovermode='closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='scatter2010')

# Scatter plot: Extra Trees feature importances
trace = go.Scatter(
    x=feature_dataframe['features'].values,
    y=feature_dataframe['Extra Trees feature importances'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe['Extra Trees feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Extra Trees feature importances',
    hovermode='closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='scatter2011')

# Scatter plot: AdaBoost feature importances
trace = go.Scatter(
    x=feature_dataframe['features'].values,
    y=feature_dataframe['AdaBoost feature importances'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='AdaBoost feature importances',
    hovermode='closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='scatter2012')

# Scatter plot: Gradient Boost feature importances
trace = go.Scatter(
    x=feature_dataframe['features'].values,
    y=feature_dataframe['Gradient Boost feature importances'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=25,
        #       size= feature_dataframe['AdaBoost feature importances'].values,
        # color = np.random.randn(500), #set color equal to a variable
        color=feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text=feature_dataframe['features'].values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Gradient Boost feature importances',
    hovermode='closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='scatter2013')

# Calculate the mean of all the importances and store it as a new column of the
feature_dataframe['mean'] = feature_dataframe.mean(axis=1)
print(feature_dataframe)

x = feature_dataframe['features'].values
y = feature_dataframe['mean'].values

data = [
    go.Bar(
        x=x,
        y=y,
        width=0.5,
        marker=dict(
            color=feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )
]

layout = go.Layout(
    autosize=True,
    title='Barplots of Mean Feature importance',
    hovermode='closest',
    yaxis=dict(
        title='Feature Importance',
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)

fig = go.Figure(data, layout)
py.plot(fig, filename='bar-direct-labels')

# Building a dataframe from the first level features to be used in the second level predictions

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel()
                                       })

X_train_second = np.concatenate((
    et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train
), axis=1)

X_test_second = np.concatenate((
    et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test
), axis=1)

gbm = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1
).fit(X_train_second, y_train)

predictions = gbm.predict(X_test_second)

print(predictions)

StackingSubmission = pd.DataFrame({
    'PassengerId': PassengerId,
    'Survived': predictions
})

StackingSubmission.to_csv("StackingSubmission.csv", index=False)
