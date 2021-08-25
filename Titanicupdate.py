#!/usr/bin/env python
# coding: utf-8

# # Titanic survival prediction 
# 
# This notebook goes through a basic exploratory data analysis of the Kaggle dataset with Python

# ## 1) Importing relevant depedencies 

# In[6]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')

# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# In[7]:


conda install -c conda-forge catboost


# In[8]:


#Importing train and test data

train = pd.read_csv("train.csv")
test = pd.read_csv("ttest.csv")
gender_submission = pd.read_csv("gender_submission.csv")


# In[9]:


gender_submission.head()


# In[10]:


train.head()


# In[11]:


test.head()


# In[12]:


len(train)


# In[13]:


len(test)


# In[14]:


train.describe()


# ## Checking out missing values

# In[15]:


missingno.matrix(train, figsize=(30,10))


# In[16]:


train.isnull().sum()


# ## Creating two new dataframes for data analysis
# 
# We'll create one for exploring discretised continuous variables (continuous variables which have been sorted into some kind of category) and another for exploring continuous variables.

# In[17]:


df_bin = pd.DataFrame() # for discretised continuous variables 0-10, 10-20
df_con = pd.DataFrame() # for continuous variables 


# In[18]:


# Different data types in the dataset
train.dtypes


# ## Exploring each feature individually  

# ## Target feature : Survived
# 
# Key: 0: Not survied , 1: Survived
# 
# This is the feature we want to predict based off all the others.

# In[19]:


#How many survived 

fig,ax = plt.subplots(figsize=(20,2))
sns.countplot(y= "Survived", data = train)
print(train.Survived.value_counts())


# In[20]:


#Adding this to our sub dataframes

df_bin["Survived"] = train["Survived"]
df_con["Survived"] = train["Survived"]


# In[21]:


df_bin.head()


# In[22]:


df_con.head()


# ## Feature: Pclass
# 
# Description : Ticket class of a passenger
# 
# Key: 1=1st , 2= 2nd , 3= 3rd 
# 
# * ordinal variable 
# 
# 

# In[23]:


sns.distplot(train.Pclass) #Exploring the spread of values


# We can see that the data in this feature is numerical yet they are categories

# In[24]:


df_bin["Pclass"] = train["Pclass"] #Adding Pclass directly to sub dataframes as there are no missing values
df_con["Pclass"] = train["Pclass"] 


# In[25]:


df_bin.head()


# ## Feature : Name
# 
# Description : The name of the passenger 

# In[26]:


#To check different names 

train.Name.value_counts()[:50]


# In[27]:


len(train)


# ## Feature: Sex
# 
# Description : The sex of the passenger
# 

# In[28]:


#Distribution of sex

ax,fig = plt.subplots(figsize=(20,5))
sns.countplot(y="Sex", data = train);


# Since this data is already binary , we can add it to our sub dataframes.

# In[29]:


df_bin["Sex"] = train["Sex"]

df_bin["Sex"] = np.where(df_bin["Sex"]=="female", 1,0)

df_con["Sex"] = train["Sex"]


# In[30]:


df_bin.head()


# In[31]:


#How does the Sex variable look compared to Survival

fig,ax  = plt.subplots(figsize=(10,10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'});
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'}); #blue- survived, yellow = not survive
plt.legend()


# More number of females survived than number of men.

# In[32]:


train.isnull().sum()


# ## Feature : Age
# 
# Description: Age of passenger
# 
# Out of 891 , 177 values are missing 

# In[33]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_imputer = SimpleImputer(strategy = "median")

num_feature = ["Age"]

imputer= ColumnTransformer([("num_imputer", num_imputer, num_feature)])

#Transform the data 
filled_age = imputer.fit_transform(train)

filled_age


# In[34]:


Age_test = imputer.fit_transform(test)


# In[35]:


train["AGE"] = filled_age
test["Age_test"] = Age_test


# In[36]:


train.head()


# In[37]:


test.head()


# In[38]:


train = train.drop("Age", axis =1) # Dropping previous age columns which had missing values


# In[39]:


test = test.drop("Age", axis =1)


# In[40]:


train.isnull().sum()


# In[41]:


df_con["Age"] = pd.cut(train["AGE"],10) # bucketed into different values
df_bin["Age"] = train["AGE"] #non bucketed values


# In[42]:


df_con.head()


# In[43]:


df_bin.head()


# In[44]:


fig,ax  = plt.subplots(figsize=(10,10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Age'], kde_kws={'label': 'Survived'});
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Age'], kde_kws={'label': 'Did not Survive'});

plt.legend()


# # Function to create count and distribution plots

# In[45]:


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
        plt.legend()
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"});
        plt.legend()


# ## Feature : SibSp
# 
# Description : No. of spouses/siblings the passenger has aboard the titanic
# 
# 

# In[46]:


train.SibSp.value_counts()


# In[47]:


#Add SibSp to our subset Dataframes

df_bin["SibSp"] = train["SibSp"]
df_con["SibSp"] = train["SibSp"]


# In[48]:


#Visualize the counts of SibSp and distribution of values against Survived

plot_count_dist(train,bin_df= df_bin, label_column="Survived", target_column = "SibSp", figsize = (20,10))


# ## Feature Parch
# 
# Description : The number of parents/children the passenger has aboard the Titanic.
# This feature is similar to SibSp
# 
# 

# In[49]:


#Visualize the counts of Parch and distribution of the values
#Against Survived

plot_count_dist(train,bin_df=df_bin,label_column="Survived", target_column = "Parch",figsize=(20,10))


# In[50]:


train.Parch.value_counts()


# In[51]:


#Adding Parch to our subset dataframes

df_bin["Parch"] = train["Parch"]

df_con["Parch"] = train["Parch"]


# In[52]:


df_con.head()


# In[53]:


train.head()


# ## Feature: Ticket
# 
# Description : Ticket number of boarding passenger 
# 
# 

# In[54]:


train.Ticket.value_counts()


# In[55]:


len(train.Ticket.unique()) #681 unique values


# ## Feature: Fare 
# 
# Description : The cost of ticket

# In[56]:


sns.countplot(y= "Fare", data = train);


# In[57]:


len(train.Fare.unique()) #There are 248 unique values


# In[58]:


df_con["Fare"] = train["Fare"] 

df_bin["Fare"] = pd.cut(train["Fare"], bins=5) #discretised


# In[59]:


df_bin.head()


# In[60]:


df_con.head()


# In[61]:


df_bin.Fare.value_counts()


# In[62]:


plot_count_dist(data= train, bin_df = df_bin,label_column = "Survived", target_column = "Fare", figsize =(20,10),use_bin_df= True )


# ## Feature : Cabin
# 
# Description : The cabin number where the passenger was staying 

# In[63]:


train.isnull().sum()


# We will not be using Cabin feature as it has so many missing values

# ## Feature : Embarked
# 
# Description: The port where the passenger boarded the Titanic 
# 
# C: Cherbourg Q = Queenstown S = Southampton 

# In[64]:


train.isnull().sum()


# In[65]:


train.Embarked.value_counts()


# In[66]:


sns.countplot(y= "Embarked", data =train)


# In[67]:


#Adding Embarked feature to sub dataframes

df_bin["Embarked"] = train["Embarked"]  
df_con["Embarked"] = train["Embarked"]


# In[68]:


#Removing rows from Embarked feature having missing values
print(len(df_con))
df_con = df_con.dropna(subset=["Embarked"])
df_bin= df_bin.dropna(subset=["Embarked"])
print(len(df_con))


# In[69]:


df_bin.head()


# In[70]:


df_con.head()


# In[71]:


train.head()


# # Feature Encoding
# 
# Now we have our two sub dataframes ready. We can encode the features so they're ready to be used with our machine learning models.
# 
# 

# In[72]:


# One-hot encode binned variables
one_hot_cols = df_bin.columns.tolist()
one_hot_cols.remove('Survived')
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)

df_bin_enc.head()


# In[73]:



# One hot encode the categorical columns
df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_one_hot = pd.get_dummies(df_con['Sex'], 
                                prefix='sex')

df_plcass_one_hot = pd.get_dummies(df_con['Pclass'], 
                                   prefix='pclass')


# In[74]:


# Combine the one hot encoded columns with df_con_enc
df_con_enc = pd.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot, 
                        df_plcass_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)


# In[75]:


# Let's look at df_con_enc
df_con_enc.head(20)


# ## Building ML models

# ### Separating the data

# In[76]:


#Select the dataframe we want to use

selected_df = df_con_enc


# In[77]:


df_con_enc = df_con_enc.drop("Age", axis = 1)


# In[78]:


df_con_enc["Age"] = train["AGE"]


# In[79]:


selected_df.drop("Age", axis =1)


# In[80]:


selected_df["Age"] =  df_con_enc["Age"]


# In[81]:


X_train = selected_df.drop("Survived" , axis =1)

y_train = selected_df["Survived"]


# In[82]:


X_train.shape


# In[83]:


y_train.shape


# In[84]:


def fit_ml_algo(algo, X_train, y_train, cv):
    
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train,y_train) *100,2)
    
    train_pred = model_selection.cross_val_predict(algo, X_train, y_train, cv=cv, n_jobs = -1)
    
    acc_cv = round(metrics.accuracy_score(y_train, train_pred)*100,2)
    
    return train_pred, acc ,acc_cv


# In[85]:


# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               y_train, 
                                                                    10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# In[86]:



# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  y_train, 
                                                  10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))


# In[87]:


# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 
                                                                      X_train, 
                                                                      y_train, 
                                                                           10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))


# In[88]:



# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))


# In[89]:



# Stochastic Gradient Descent
start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(), 
                                                  X_train, 
                                                  y_train,
                                                  10)
sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))


# In[90]:



# Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))


# In[91]:


# Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(), 
                                                                       X_train, 
                                                                       y_train,
                                                                       10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))


# ## Catboost Algorithm

# In[92]:


X_train.head()


# In[93]:


y_train.head()


# In[94]:


X_train.head()


# In[95]:


X_train["Age"] = X_train["Age"].astype(int)


# In[96]:


# Define the categorical features for the CatBoost model
cat_features = np.where(X_train.dtypes != np.float)[0]
cat_features


# In[97]:


# Use the CatBoost Pool() function to pool together the training data and categorical feature labels
train_pool = Pool(X_train, 
                  y_train,
                  cat_features)


# In[98]:


#Catboost model definition 

catboost_model = CatBoostClassifier(iterations = 1000, custom_loss= ["Accuracy"], loss_function = "Logloss")

#Fit Catboost model 

catboost_model.fit(train_pool,plot = True)

#Catboost accuracy

acc_catboost = round(catboost_model.score(X_train,y_train)*100,2)


# In[99]:


# How long will this take?
start_time = time.time()

# Set params for cross-validation as same as initial model
cv_params = catboost_model.get_params()

# Run the cross-validation for 10-folds (same as the other models)
cv_data = cv(train_pool,
             cv_params,
             fold_count=10,
             plot=True)

# How long did it take?
catboost_time = (time.time() - start_time)

# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)


# In[100]:


# Print out the CatBoost model metrics
print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))
print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))


# ## Model Results

# In[101]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_knn, 
        acc_log,  
        acc_gaussian, 
        acc_sgd, 
        acc_linear_svc, 
        acc_dt,
        acc_gbt,
        acc_catboost
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)


# In[102]:



cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_cv_knn, 
        acc_cv_log,      
        acc_cv_gaussian, 
        acc_cv_sgd, 
        acc_cv_linear_svc, 
        acc_cv_dt,
        acc_cv_gbt,
        acc_cv_catboost
    ]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)


# ## Feature importance

# In[103]:


# Feature Importance
def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    return fea_imp
    #plt.savefig('catboost_feature_importance.png')


# In[104]:


# Plot the feature importance scores
feature_importance(catboost_model, X_train)


# In[105]:



metrics = ['Precision', 'Recall', 'F1', 'AUC']

eval_metrics = catboost_model.eval_metrics(train_pool,
                                           metrics=metrics,
                                           plot=True)

for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))


# In[106]:



# We need our test dataframe to look like this one
X_train.head()


# In[107]:


# Our test dataframe has some columns our model hasn't been trained on
test.head()


# In[108]:


# One hot encode the columns in the test data frame (like X_train)
test_embarked_one_hot = pd.get_dummies(test['Embarked'], 
                                       prefix='embarked')

test_sex_one_hot = pd.get_dummies(test['Sex'], 
                                prefix='sex')

test_plcass_one_hot = pd.get_dummies(test['Pclass'], 
                                   prefix='pclass')


# In[109]:


# Combine the test one hot encoded columns with test
test = pd.concat([test, 
                  test_embarked_one_hot, 
                  test_sex_one_hot, 
                  test_plcass_one_hot], axis=1)


# In[110]:


test.head()


# In[111]:


# Create a list of columns to be used for the predictions
wanted_test_columns = X_train.columns
wanted_test_columns


# In[112]:


test = test.rename(columns = {"Age_test": "Age"})


# In[113]:


test["Age"] = test["Age"].astype(int)


# In[114]:


test.head()


# In[115]:


# Make a prediction using the CatBoost model on the wanted columns
predictions = catboost_model.predict(test[wanted_test_columns])


# In[116]:


# Our predictions array is comprised of 0's and 1's (Survived or Did Not Survive)
predictions[:20]


# In[117]:


submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions # our model predictions on the test dataset
submission.head()


# In[118]:


# What does our submission have to look like?
gender_submission.head()


# In[119]:


# Are our test and submission dataframes the same length?
if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")


# In[120]:


# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('../kaggle_submission.csv', index=False)
print('Submission CSV is ready!')


# In[121]:


# Check the submission csv to make sure it's in the right format
submissions_check = pd.read_csv("../kaggle_submission.csv")
submissions_check.head()


# In[122]:


import pickle


# In[123]:


pickle.dump(catboost_model, open("model.pkl", "wb"))


# In[124]:


model = pickle.load(open("model.pkl", 'rb'))


# In[125]:


print(model.predict([[52,0,0,9.8,0,1,1,1,0,1,0,0]]))


# In[ ]:




