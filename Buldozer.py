# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:32:42 2018

@author: ChrisB
"""
# to handle datasets
import pandas as pd
import numpy as np


# for plotting
import matplotlib.pyplot as plt

from matplotlib import pyplot

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import xgboost as xgb

# to evaluate the models
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import re

from pandas.tools.plotting import scatter_matrix
import seaborn as sns
sns.set_color_codes()
sns.set(font_scale=1.25)

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# feature scaling
from sklearn.preprocessing import StandardScaler


import xgboost as xgb

# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

pd.pandas.set_option('display.max_columns', None)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings('ignore')



# loading the Training  dataset 1
data1 = pd.read_csv('Train.csv')


# Examing the first 5 rows of the dataset
#print(data1.head())

# Determining the dimensions of the data
#print(data1.shape)

# loading Training dataset 2
data2 = pd.read_csv('Machine_Appendix.csv')
# Extracting relevant variables from dataset 2
dataset2_rel = ['MachineID','ModelID', 'PrimarySizeBasis']
data2_rev = data2[dataset2_rel]
# Merging datasets
data_new1 = pd.merge(data1,data2_rev, how='left', on=['MachineID', 'ModelID'])
# load test dataset 1
data_test = pd.read_csv('Test.csv')
# merging test data with machine_appendix data
data_test_mer = pd.merge(data_test,data2_rev, how='left', on=['MachineID', 'ModelID'])
# Extracting the year from a time object
data_new1['saledate'] = pd.to_datetime(data_new1['saledate'])
#use the datetime attribute dt to access the datetime components
data_new1['yearsold'] = data_new1['saledate'].dt.year

# Creating an Age variable
data_new1['MachineAge'] = data_new1['yearsold'] - data_new1['YearMade']
# Extracting the year from a time object
data_test_mer['saledate'] = pd.to_datetime(data_test_mer['saledate'])
#use the datetime attribute dt to access the datetime components
data_test_mer['yearsold'] = data_test_mer['saledate'].dt.year
# Creating an Age variable for the Test dataset
data_test_mer['MachineAge'] = data_test_mer['yearsold'] - data_test_mer['YearMade']

# let's visualise the percentage of missing values in the Training dataset
for var in data_new1.columns:
     if data_new1[var].isnull().sum()>0:
        print(var, data_new1[var].isnull().mean())
       
# let's inspect the type of those variables with a lot of missing information in the Training dataset
for var in data_new1.columns:
    if data_new1[var].isnull().mean()>0.4:
        print(var, data_new1[var].unique())

        

# let's visualise the percentage of missing values in the Test dataset
for var in data_test_mer.columns:
    if data_test_mer[var].isnull().sum()>0:
       print(var, data_test_mer[var].isnull().mean())
       

# let's inspect the type of those variables with a lot of missing information in the Test dataset
for var in data_test_mer.columns:
    if data_test_mer[var].isnull().mean()>0.1:
        print(var, data_test_mer[var].unique())

# Deleting meaningless variables from the Training dataset
data_new1.drop(['yearsold','MachineID','ModelID','datasource','auctioneerID','YearMade','saledate','fiModelDesc','fiModelDescriptor','fiProductClassDesc'], axis=1,inplace=True)

#Deleting meaningless variables from the Test dataset
data_test_mer.drop(['yearsold','MachineID','ModelID','datasource','auctioneerID','YearMade','saledate','fiModelDesc','fiModelDescriptor','fiProductClassDesc'], axis=1,inplace=True)
X_train = data_new1.copy()

type(X_train)
X_test = data_test_mer.copy()

type(X_test)


# find categorical variables in the Training set
categorical_train = [var for var in X_train.columns if X_train[var].dtype=='O']
print('There are {} categorical variables in the Training set'.format(len(categorical_train)))
# find numerical variables
numerical_train = [var for var in X_train.columns if X_train[var].dtype!='O']
print('There are {} numerical variables in the Training set'.format(len(numerical_train)))
# find categorical variables in the Test set
categorical_test = [var for var in X_test.columns if X_test[var].dtype=='O']
print('There are {} categorical variables in the Test set'.format(len(categorical_test)))


# find numerical variables in the Test set
numerical_test = [var for var in X_test.columns if X_test[var].dtype!='O']
print('There are {} numerical variables in the Test set'.format(len(numerical_test)))

continuous_train = [var for var in numerical_train if var not in [ 'SalesID','SalePrice']]
continuous_train


continuous_test = [var for var in numerical_test if var not in [ 'SalesID','SalePrice']]
continuous_test


# let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution in the Training set

for var in continuous_train:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = X_train.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = X_train[var].hist(bins=20)
    fig.set_ylabel('Number of Buldozers')
    fig.set_xlabel(var)

    plt.show()
    
    # let's make boxplots to visualise outliers in the continuous variables 
# and histograms to get an idea of the distribution in the Test set

for var in continuous_test:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = X_test.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = X_test[var].hist(bins=20)
    fig.set_ylabel('Number of Buldozers')
    fig.set_xlabel(var)

    plt.show()
    
    
 # Examining the number of labels for the categorical variables in the Training set
for var in categorical_train:
    print(var, ' contains ', len(X_train[var].unique()), ' labels')
# Examining the number of labels for the categorical variables in the Test set
for var in categorical_test:
    print(var, ' contains ', len(X_test[var].unique()), ' labels')
    
# print variables with missing data in Training set
for col in continuous_train:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# print variables with missing data in Test set
for col in continuous_test:
    if X_test[col].isnull().mean()>0:
        print(col, X_test[col].isnull().mean())
        
# MachineHoursCurrentMeter 
extreme = X_train.MachineHoursCurrentMeter.mean() + X_train.MachineHoursCurrentMeter.std()*3
for df in [X_train, X_test]:
    df.MachineHoursCurrentMeter.fillna(extreme, inplace=True)

# print variables with missing data in Training set
for col in categorical_train:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
        
# Imputing missing data in Training and Test sets with the word "Missing"
for col in categorical_train:
    if X_train[col].isnull().mean()<.08:
        for df in [X_train, X_test]:
            df[col].fillna(X_train[col].mode()[0], inplace=True) 
    else:
        for df in [X_train, X_test]:
            df[col].fillna('Missing', inplace=True)

# check absence of null values
X_train.isnull().sum()

# Checking the test set for missing values
X_test.isnull().sum()

# Defining function to flag categorical label with low proportion of respondents as "rare"
def rare_imputation(variable, which='rare'):    
    # find frequent labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)
        
    else:
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
        
#Imputing "rare" for rare labels
for col in categorical_train:
    rare_imputation(col, 'rare')

#Imputing "frequent" for rare labels
for col in categorical_train:
    rare_imputation(col, 'frequent')
    
# Checking result of imputation
for col in categorical_train:
    print(X_train[col].value_counts()/np.float(len(X_train)))
    
# Checking result of imputation - Test set
for col in categorical_test:
    print(X_test[col].value_counts()/np.float(len(X_test)))
    

y_train = X_train['SalePrice']
    
def tree_binariser(var):
    score_ls = [] # here I will store the mse

    for tree_depth in [1,2,3,4]:
        # call the model
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)

        # train the model using 3 fold cross validation
        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, cv=3, scoring='neg_mean_squared_error')
        score_ls.append(np.mean(scores))

    # find depth with smallest mse
    depth = [1,2,3,4][np.argmax(score_ls)]
    #print(score_ls, np.argmax(score_ls), depth)

    # transform the variable using the tree
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)
    X_train[var] = tree_model.predict(X_train[var].to_frame())
    X_test[var] = tree_model.predict(X_test[var].to_frame())
    
continuous_train
continuous_test


for var in continuous_train:
    tree_binariser(var)

for var in continuous_test:
    tree_binariser(var)
    

X_train[continuous_train].head()

for var in continuous_train:
    print(var, len(X_train[var].unique()))
    
X_test[continuous_test].head()

for var in continuous_test:
    print(var, len(X_test[var].unique()))
    
def encode_categorical_variables(var, target):
        # make label to price dictionary
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        # encode variables
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        

# encode labels in categorical vars
for var in categorical_train:
    encode_categorical_variables(var, 'SalePrice')


#let's inspect the dataset
X_train.head()

X_train.describe()

# Taking a Random Sample of 10000 observations from the x_Train
# you can use random_state for reproducibility
X_train = X_train.sample(n=10000, random_state=2)

# Taking a Random Sample of 10000 observations from the x_Test
# you can use random_state for reproducibility
X_test = X_test.sample(n=10000, random_state=2)

#Checking the sample
X_train.head()


#Checking the sample
X_test.head()

# Change the row index
X_test.index = range(10000)


X_test.head()


X_test.shape

# Splitting the Training set in label and features
y_train = X_train['SalePrice']


X_train1 = X_train.copy()
X_test1 = X_test.copy()


type(X_train1),type(X_test1)

X_train1.head(1)


X_test1.head(1)


y_train.shape

# Removing the label/outcome variable from the data sets
X_train.drop(labels=['SalesID', 'SalePrice'], inplace=True, axis=1)
X_test.drop(labels=['SalesID'], inplace=True, axis=1)


# Checking if the change worked
X_train.head(2)


# Checking if the change worked
X_test.head(2)

# I keep a copy of the dataset with all the variables
# to measure the performance of machine learning models
# at the end of the notebook
 
X_train_original = X_train.copy()
X_test_original = X_test.copy()


X_train_original.shape

X_train_original.head()

#============================================================================================================
from sklearn.pipeline import Pipeline
import sklearn.grid_search
select = sklearn.feature_selection.SelectKBest(k=10)
print(select)
clf = sklearn.ensemble.RandomForestRegressor()
steps = [('feature_selection', select),
        ('random_forest', clf)]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
### call pipeline.predict() on your X_test data to make a set of test predictions
pred = pipeline.predict(X_train)
### test your predictions using sklearn.classification_report()
print('========================================================================================================')
print('rf train mse: {}'.format(mean_squared_error( y_train, pred)))
coefficient_of_dermination = r2_score( y_train, pred)
print('Rsquare : ', coefficient_of_dermination)
print('====================================================')

#Following code will help you in finding top K features with their F-scores. Let, X is the pandas dataframe, whose columns are all the features and y is the list of class labels.
from sklearn.feature_selection import SelectKBest, f_classif
#Suppose, we select 10 features with top 5 Fisher scores
selector = SelectKBest(f_classif, k = 10)
#New dataframe with the selected features for later use in the classifier. fit() method works too, if you want only the feature names and their corresponding scores
X_new = selector.fit_transform(X_train, y_train)
names = X_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)


print('==================================================================================')

parameters = dict(feature_selection__k=[5, 44], 
              random_forest__n_estimators=[50, 100, 200],
              random_forest__min_samples_split=[2, 3, 4, 5, 10])

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
pred = cv.predict(X_train)

print('========================================================================================================')
print('rf train mse: {}'.format(mean_squared_error( y_train, pred)))
coefficient_of_dermination = r2_score( y_train, pred)
print('Rsquare : ', coefficient_of_dermination)
print('====================================================')

print("Best score: %0.3f" % cv.best_score_)
print("Best parameters set:")
best_parameters = cv.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print('================================================================================')
features = pipeline.named_steps['feature_selection']
print(X_train.columns[features.get_support()])























