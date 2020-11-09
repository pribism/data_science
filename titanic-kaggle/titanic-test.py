#Data Science Project for the first Kaggle Dataset: titanic (https://www.kaggle.com/c/titanic/notebooks)
#Written based on the https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

#Initial imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#Read the data
train = pd.read_csv('./titanic_train.csv', index_col = 0)
test = pd.read_('./titanic_test.csv', index_col = 0)

PassengerId = test['PassengerId']

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
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

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

#Create a feature CategoricalFare by cutting the Fare in 4 Percentile ranges
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)


#Build a linear regression model
# Train and Split the train dataset
X =
y =
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

train = pd.get_dummies(train, columns = [''], drop_first = True)