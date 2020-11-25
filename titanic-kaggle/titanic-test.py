#Data Science Project for the first Kaggle Dataset: titanic (https://www.kaggle.com/c/titanic/notebooks)
#Written based on the https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

#Initial imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn
#import xgboost as xgb
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

import SklearnHelper.py as skh

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

#Read the data
train = pd.read_csv('./titanic_train.csv', index_col = 0)
test = pd.read_csv('./titanic_test.csv', index_col = 0)

#Show all the columns
pd.set_option('display.max_columns', None)

#Feature engineering part
full_data = [train, test]

#Computes the length of the name of a person
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

#True, if the person has a cabin
train['Has_cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

#Fix the missing Age by putting the avg age of class
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
    dataset.loc[:, 'Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset.loc[:, 'Age'] = dataset['Age'].astype(int)

#Create a New Feature CategoricalAge
train['CategoricalAge'] = pd.cut(train['Age'], 5)

#Fix the missing Embark by choosing the most common Embark
#Fill the missing Embarked with the most common value
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Create new Feature Family Size as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Create a feature: is alone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#Fills the Empty fields in Fare column with median and creates a feature CategoricalFare by cutting the Fare in 4 Percentile ranges
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

#Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""

#Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#Group all non-common titles into one single group "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train['Fare'].describe())
print(train.dtypes)


#Mapping part (get dummy data)
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    #Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    #Mapping Titles
    dataset['Title'] = dataset['Title'].map( {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} )
    dataset['Title'] = dataset['Title'].fillna(0)

    #Mapping Fare (try using cut in the future?)
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.545), 'Fare'] = 1
    dataset.loc[ (dataset['Fare'] > 14.545) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    #Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

#Feature selection
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test = test.drop(drop_elements, axis = 1)

#Test: Print the resulting dataframe
#print(train.head(3))

colormap = plt.cm.RdBu
plt.figure('Pearson Correlation of Features', figsize = (14, 12))
plt.title('Pearson Correlation of Features')
sns.heatmap(train.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
            square = True, cmap = colormap, linecolor= 'white', annot= True)
plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5 # sets folds for out of fold prediction
kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)

    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain, ))
        oof_test = np.zeros((ntest, ))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

#Build a linear regression model
# Train and Split the train dataset
#X =
#y =
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3
#)

#train = pd.get_dummies(train, columns = [''], drop_first = True)

