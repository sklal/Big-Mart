# Problem Statement
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

# Libraries Used
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the test and train data

import os
dirpath = os.getcwd()
print("current directory is : " + dirpath)
# Train
filepath1 = os.path.join(os.getcwd(), 'Train.csv')
train = pd.read_csv(filepath1)
# Test
filepath2 = os.path.join(os.getcwd(), 'Test.csv')
test = pd.read_csv(filepath2)

# Combining both Train & Test data

train['source'] = 'train'
test['source'] = 'test'
combined = pd.concat([train, test])

combined.head()

# Performing Basic Checks -


def dfChkBasics(dframe):
    cnt = 1

    try:
        print(str(cnt)+': info(): ')
        cnt += 1
        print(dframe.info())
    except:
        pass

    print(str(cnt)+': describe(): ')
    cnt += 1
    print(dframe.describe())

    print(str(cnt)+': dtypes: ')
    cnt += 1
    print(dframe.dtypes)

    try:
        print(str(cnt)+': columns: ')
        cnt += 1
        print(dframe.columns)
    except:
        pass

    print(str(cnt)+': head() -- ')
    cnt += 1
    print(dframe.head())

    print(str(cnt)+': shape: ')
    cnt += 1
    print(dframe.shape)


def dfChkValueCnts(dframe):
    for i in dframe.columns:
        print(dframe[i].value_counts())


dfChkBasics(combined)
dfChkValueCnts(combined)

combined.dtypes
# Fixing the data types

combined['Item_Weight'] = combined['Item_Weight'].astype('float64')
combined.dtypes

# Fixing the Item Fat Content column

combined.loc[combined['Item_Fat_Content'].str.contains(
    "LF|low fat"), 'Item_Fat_Content'] = 'Low Fat'
combined.loc[combined['Item_Fat_Content'].str.contains(
    "reg"), 'Item_Fat_Content'] = 'Regular'

combined['Item_Fat_Content'].value_counts()

# Categorizing Item Type into fewer Categories

# Taking the first two alphabets of the item identifier to compartmentalize the item type
combined['Item_Type_Combined'] = combined['Item_Identifier'].apply(
    lambda x: x[0:2])

# Standardizing the names
combined['Item_Type_Combined'] = combined['Item_Type_Combined'].map({'FD': 'Food',
                                                                     'NC': 'Non-Consumable',
                                                                     'DR': 'Drinks'})

combined['Item_Type_Combined'].value_counts()

# Mark non-consumables as separate category in low_fat:
combined.loc[combined['Item_Type_Combined'] ==
             "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
combined['Item_Fat_Content'].value_counts()

# Checking for missing values
combined.apply(lambda x: sum(x.isnull()))

#  Item_Outlet_Sales is the target variable and missing values and is present in the test set

# Handling missing values

# Filling the null values in weight by the mean of the column

combined['Item_Weight'].fillna(combined['Item_Weight'].mean(), inplace=True)


combined['Item_Weight']

combined.dtypes
# Filling the null values in outlet size by the mode of the column

combined['Outlet_Size'].fillna(combined['Outlet_Size'].mode()[0], inplace=True)


# After handling missing values
combined.apply(lambda x: sum(x.isnull()))


# Univariate Analysis

plt.figure(figsize=(30, 8))
Itemtype = combined['Item_Type_Combined'].value_counts()
plt.title('Distribution of Item Categories')
sns.barplot(x=Itemtype[:15].keys(),
            y=Itemtype[:15].values, palette="GnBu_d")

plt.figure(figsize=(10, 8))
FatContent = combined['Item_Fat_Content'].value_counts()
plt.title('Distribution of Items based on Fat Content')
sns.barplot(x=FatContent[:15].keys(),
            y=FatContent[:15].values, palette="GnBu_d")

plt.figure(figsize=(10, 8))
Outlettype = combined['Outlet_Type'].value_counts()
plt.title('Distribution of Items based on Fat Content')
sns.barplot(x=Outlettype[:15].keys(),
            y=Outlettype[:15].values, palette="GnBu_d")

plt.figure(figsize=(10, 8))
Loctype = combined['Outlet_Location_Type'].value_counts()
plt.title('Distribution of Stores across Locations')
sns.barplot(x=Loctype[:15].keys(),
            y=Loctype[:15].values, palette="GnBu_d")

combined['Outlet_Size'].value_counts()
plt.figure(figsize=(10, 8))
Outsize = combined['Outlet_Size'].value_counts()
plt.title('Distribution of Outlet Size')
sns.barplot(x=Outsize[:15].keys(),
            y=Outsize[:15].values, palette="GnBu_d")

combined.columns
combined.dtypes

# Numerical Features

numeric_features_combined = combined.select_dtypes(include=[np.number])
numeric_features_combined.dtypes

# Let's see the correlation between Numerical predictors and Target Variable
corr_combined = numeric_features_combined.corr()
corr_combined

# Correlation with respect to our Target Variable

print(corr_combined['Item_Outlet_Sales'].sort_values(ascending=False))

# correlation matrix
ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_combined, vmax=.8, square=True, cmap="Blues")

# Coversion of Nominal Categorical variable into Numerical for Scikit Learn Library

le = LabelEncoder()

# New variable for outlet
combined['Outlet'] = le.fit_transform(combined['Outlet_Identifier'])

var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size',
           'Item_Type_Combined', 'Outlet_Type', 'Outlet', 'Outlet_Establishment_Year']


for i in var_mod:
    combined[i] = le.fit_transform(combined[i])


# One Hot Coding to make a column for each of nominal value in a particular column :
combined = pd.get_dummies(combined, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                             'Item_Type_Combined', 'Outlet'])

# Corresponding Values
# Item_Fat_Content0 - Low Fat
# Item_Fat_Content1 - Regular
# Item_Fat_Content2 - Non-Edible
# Outlet_Location_Type0 - Tier 1
# Outlet_Location_Type1 - Tier 2
# Outlet_Location_Type2 - Tier 3
# Outlet_Size0 - Medium
# Outlet_Size1 - Small
# Outlet_Size2 - High
# Item_Type_Combined0 - Food
# Item_Type_Combined1 - Non-Consumable
# Item_Type_Combined2 - Drinks
# Outlet_Type0 - Supermarket Type 1
# Outlet_Type1 - Grocery Store
# Outlet_Type2 - Supermarket Type 3
# Outlet_Type3 - Supermarket Type 2


combined.columns
combined.dtypes

# Coverting the dummy columns to float

for col in ['Item_Weight', 'Item_Fat_Content_0',
            'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_0',
            'Outlet_Location_Type_1', 'Outlet_Location_Type_2', 'Outlet_Size_0',
            'Outlet_Size_1', 'Outlet_Size_2', 'Outlet_Type_0', 'Outlet_Type_1',
            'Outlet_Type_2', 'Outlet_Type_3', 'Item_Type_Combined_0',
            'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_0', 'Outlet_1',
            'Outlet_2', 'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7',
            'Outlet_8', 'Outlet_9']:
    combined[col] = combined[col].astype('float64')


# #Drop the columns which have been converted to different types:
combined.drop(['Item_Type'], axis=1, inplace=True)

# Dividing into test and train:
train = combined.loc[combined['source'] == "train"]
test = combined.loc[combined['source'] == "test"]

# Dropping unnecessary columns:
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)


# Export files as modified versions:
train.to_csv(os.path.join("train_modified.csv"))
test.to_csv(os.path.join("test_modified.csv"))

train
combined
# Creating a function for model fitting
# Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']


def modelfit(alg, dtrain, dtest, predictors, target, IDcol):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Perform cross-validation:
    cv_score = cross_val_score(
        alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(
        dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" %
          (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])


predictors = [x for x in train.columns if x not in [target]+IDcol]
LR = LinearRegression(alpha=0.05, normalize=True)
modelfit(LR, train, test, predictors, target, IDcol)
coef1 = pd.Series(LR.coef_, predictors).sort_values()
coef1
coef1.plot(kind='bar', title='Model Coefficients')


# Model Report
# RMSE : 1127
# CV Score : Mean - 1129 | Std - 43.58 | Min - 1075 | Max - 1211

# Ridge Regression
RR = Ridge(alpha=0.05, normalize=True)
modelfit(RR, train, test, predictors, target, IDcol)
coef2 = pd.Series(RR.coef_, predictors).sort_values()
coef2
coef2.plot(kind='bar', title='Model Coefficients')

# Model Report
# RMSE : 1128
# CV Score : Mean - 1130 | Std - 44.81 | Min - 1076 | Max - 1218

# Decision Tree

predictors = [x for x in train.columns if x not in [target]+IDcol]
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(DT, train, test, predictors, target, IDcol)
coef3 = pd.Series(DT.feature_importances_,
                  predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

# Model Report
# RMSE : 1059
# CV Score : Mean - 1092 | Std - 45.05 | Min - 1016 | Max - 1180


# Tuning the predictors to Item_MRP,Outlet_Type0,Outlet_5,Outlet_Establishment_Year
predictors = ['Item_MRP', 'Outlet_Type_0', 'Outlet_5']
DTm = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(DTm, train, test, predictors, target, IDcol)
coef3m = pd.Series(DTm.feature_importances_,
                   predictors).sort_values(ascending=False)
coef3m.plot(kind='bar', title='Feature Importances')

# Model Report
# RMSE : 1067
# CV Score : Mean - 1090 | Std - 43.68 | Min - 1024 | Max - 1170

# Random Forest

predictors = [x for x in train.columns if x not in [target]+IDcol]
RF = RandomForestRegressor(
    n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
modelfit(RF, train, test, predictors, target, IDcol)
coef5 = pd.Series(RF.feature_importances_,
                  predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

# Model Report
# RMSE : 1073
# CV Score : Mean - 1083 | Std - 43.99 | Min - 1019 | Max - 1160
