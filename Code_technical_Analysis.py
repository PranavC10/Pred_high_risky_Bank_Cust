# Team 1 (Data Ninja)

#### Author : Pranav Chandaliya, Vishesh Bhati, Steven Tian, Nusrat Prithee

## Problem Statement 

# Predicting high-risk customers who are likely to churn for the banking industry

### A  bank is struggling to maintain its strong foothold in the local market due to: 

###     * Rapidly increasing customer churn which leads to revenue loss for the bank
###     * The decline in overall customer base (high churn rate combined with low acquisition rate), leading to a decline in total market share. 
###     * Bank want to retain customer to stop revenue loss

#### Data Source : Kaggle

# Data Description :
### * RowNumber : Row number
### * CustomerId : Unique Customer ID 
### * Surname: Last Name of the Customer
### * CreditScore: Credit score of the customer
### * Geography:  Country of the customer
### * Gender: Gender of the customer
### * Age: Current age of the customer
### * Tenure: No. Of years customer with the bank
### * Balance: Account balance of the customers
### * NumOfProduct: Number of the products used by customer with the bank
### * HasCrCard: Customer owns credit card or not
### * isActiveMember: Customer is active member ( Interaction with App/Web Bank Application)
### * Estimated Salary: Salary of the customers
### * Exited: Yes if customer churned else no

# Contents :
### * Loading of dataset, libraries and basic statistics of the data
### * EDA 
### * Data Preprocessing  
###     * Removal of columns
###     * Encoding of Categorical Varibales
### * Predictive Modeling 
###   * Testing different balancing techniques (on Logistic Regression)
###   * Function to evaluate classifier (Reduce code size and increase reusability of the code)
###   * Random Forest 
###   * Hyperparameter tunning of Random Forest
###   * Boosting model comparision  : XGBoost,CatBoost, LightGBM
###   * Hyperparameter tunning of Boosting model
###   * Hyperparameter tunning of boosting models
###   * Model interpretability using SHAP (SHapley Additive exPlanations)
###   * Understanding model prediction
###   * Sample prediction from model
###   * Efficient probablity Threshold Cut-off for better recall


# %%
# Imorting all the necessary libraries 

### Basic Data Manupulation libraries
import pandas as pd 
import numpy as np

###  Data Visualization libraries

import seaborn as sns 
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

###  Modeling and data preprocessing libraries
from sklearn.linear_model  import  LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,roc_curve, roc_auc_score



### Model interpretability library
import shap

## Extras 
import pickle as pk ## To save model / variables
from copy import deepcopy
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import os
import warnings
warnings.filterwarnings("ignore")

print("All libraries sucessfully loaded ")
# %%
# %%
## Loading the dataset 
df=pd.read_csv(r"Data\Bank_data.csv")
df.head()
df.tail()

# %%
## Shape of the data
df.shape
# %%
# %%
## Checking column types 
df.info()
# %%
# Describe 
df.describe()

# %%
## Checking null values 
df.isnull().sum()


## EDA
# %%
### Univariate Analysis 
## Plotting pie char to understand % of customer exited 
labels = ["Current Customers","Exited Customers"]
values = list(df["Exited"].value_counts()[0:2])

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels,textinfo='label+percent', values=values, pull=[0, 0.2])])
if not os.path.exists("plot_images"):
    os.mkdir("plot_images")


fig.show()
# %%

# print styling
s1 = '\033[1m' + '\033[96m'
s2 = '\033[1m' + '\033[4m' + '\033[95m'
h1 = '\033[1m' + '\033[4m' + '\033[92m'
e = "\033[0;0m"


# %%
fig = px.histogram(df, x="CreditScore", color="Exited", color_discrete_map={"Exited":"#ffc8dd"}, nbins=60, title="Impact of Credit score on customer churn rate")

fig.update_layout(
    title="Impact of Credit score on customer churn rate",
    xaxis_title="Credit Score",
    yaxis_title="Count",
    legend_title="Exited",
)

fig.show()

print(h1 + "Finding:" + e + s1 + "Credit score in general does not have an impact on the churn rate, but in the grpah we can see that Credit score and Churn rate are normally distributed and the maximum population have credit score between 600-700 and this population has highest churn rate as well. Also we can see that churn rate is directely proportional to population. Greater the population in the bin greatr is the churn rate" + e)

# %%
#2 Balance vs churn rate

fig = px.box(df, x="Balance", color="Exited", color_discrete_map={"Exited":"#ffc8dd"}, title="Impact of Balance on churn rate")

fig.update_layout(
    title="Impact of Customer Balance on churn rate",
    xaxis_title="Customer Balance",
    yaxis_title="Count",
    legend_title="Exited"
    
)

fig.show()

print(h1 + "Finding:" + e + s1 + "From the graph we can see that people with higher balance tend to churn more than the one with the low balance." + e)

# %%
#3 Estimated salary vs churn rate

fig = px.box(df, x="EstimatedSalary", color="Exited", color_discrete_map={"Exited":"#ffc8dd"})

fig.update_layout(
    title="Impact of Salary on churn rate",
    xaxis_title="Estimated Salary",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

print(h1 + "Finding:" + e + s1 + "From the graph we can see that Estimated Salary does not have much effect on the customer churn rate the difference is very minute. " + e)

# %%

#4 Age vs churn rate

fig = px.histogram(df, x="Age", color="Exited", color_discrete_map={"Exited":"#ffc8dd"})

fig.update_layout(
    title="Impact of Age on customer churn rate",
    xaxis_title="Age",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

print(h1 + "Finding:" + e + s1 + "The graph is right skewed for the reatined customer, but the graph looks slightly normal for the exited customers. It can be seen that customer tend to churn more between the age of 40-60." + e)
# %%
# Separate each country from Dataset
geo = df['Geography']
F = df[df['Geography'] == 'France']
G = df[df['Geography'] == 'Germany']
S = df[df['Geography'] == 'Spain']

# General description of countries
df['Geography'].describe()

churn = df['Geography']
churn.value_counts()


# Distribution of each country in the dataset 

# Separate each country into churn and not churn
churnF = F[F['Exited'] == 1]
notchurnF = F[F['Exited'] == 0]
churnG = G[G['Exited'] == 1]
notchurnG = G[G['Exited'] == 0]
churnS = S[S['Exited'] == 1]
notchurnS = S[S['Exited'] == 0]
#height = [1500,2000,2500,3000,3500,4000,4500,5000,5500]

plt.xticks(np.arange(0,1.1, step=1))
plt.hist([churnF['Exited'],notchurnF['Exited'],churnG['Exited'],notchurnG['Exited'],churnS['Exited'],notchurnS['Exited']], 
        label=['France churn','France not churn','Germany churn','Germany not churn','Spain churn','Spain not churn'],
        histtype = 'bar', rwidth = 1,bins=[0, 1],edgecolor = 'black')
plt.xlabel('Churn and not churn')
plt.ylabel('Number of People')
plt.legend(loc = 0)
plt.savefig('Country churn.png')
plt.show()

# Numbers of churn by country
print('Total Numbers of France customers churned',len(churnF))
print('Total Numbers of Germany customers churned',len(churnG))
print('Total Numbers of Spain customers churned',len(churnS))

# Churn rate of each country
print('France churn rate',round(len(churnF)/len(F)*100,2))
print('Germany churn rate',round(len(churnG)/len(G)*100,2))
print('Spain churn rate',round(len(churnS)/len(S)*100,2))

print('\n')
print('Germany has the highest number of customers churned among all countries with 814 customers churned.')
print('\n')
print('Germany also has the highest churn rate of 32.44%','\n')



## Data Preprocessing 

# %%

## Dropping unwanted columns like RowNumber,CustomerID,surname
## This columns won't help in prediction of target 
df.drop(columns=["RowNumber","CustomerId","Surname"],inplace=True)
df.head(1)

# %%
## Converting Catgorical columns to numerical data using one hot encoding 
##  Geography and Gender both are nominal data hence one hot encoding would be better option
cat_col=["Geography","Gender"]
df= pd.get_dummies(df, columns=cat_col, drop_first=True)

# %%
## Spliting dataset into Training and Testing 
x=df.drop(columns=["Exited"])
y=df["Exited"]


# %%

## Base model on imbalance dataset

## We are using standard scaler as Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias.
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)
X_train, X_test, Y_train, y_test = train_test_split(X_scaled, y,stratify=y, test_size=0.25,random_state=1)


# %%
## Logistic Regression 
log_reg = LogisticRegression()

log_reg.fit(X_train,Y_train)
# %%

## Creating function that will evaluate model. 
## It will help us to reduced the code size and increase reusability of code

def evaluate_model(model,x_train,y_train,x_test,y_test,fit=False):
    '''
    Model Evaluation for Classifier
    :param  model : model object 
    :param x_train: Train features
    :param y_train: Train Target 
    :param x_test: Test features
    :param y_test: Test Target 
    :param fit bool : True if model is already fited else false

    :return: Train and Test Classification report and AUC- ROC Graph
    '''
    if fit == False:
        model.fit(x_train,y_train)
        
    
    train_pred=model.predict(x_train)
    print("Training report")
    print(classification_report(y_train, train_pred))
    
    print("Testing report")
    test_pred=model.predict(x_test)    
    print(classification_report(y_test, test_pred))

    y_pred_prob=model.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    auc=roc_auc_score(y_test,y_pred_prob[:,1])
    print("AUC Score")
    print(auc)
    
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    


    
# %%
evaluate_model(log_reg,X_train,Y_train,X_test,y_test,fit=True)
# %%
##Base model on SMOTE :
## Using SMOTE To balance Training dataset 
# transform the dataset

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(X_train, Y_train)

log_reg_smote = LogisticRegression()

log_reg_smote.fit(x_train,y_train)

print("Training data size")
print(x_train.shape)
print(y_train.value_counts())
evaluate_model(log_reg_smote,x_train,y_train,X_test,y_test,fit=True)


# %%
smt = SMOTEENN(random_state=42)
x_train, y_train = smt.fit_resample(X_train, Y_train)
x_train=pd.DataFrame(x_train,columns = x.columns)
X_test=pd.DataFrame(X_test,columns = x.columns)

log_reg_smote_ENN = LogisticRegression()

log_reg_smote_ENN.fit(x_train,y_train)

print("Training data size")
print(x_train.shape)
print(y_train.value_counts())
evaluate_model(log_reg_smote_ENN,x_train,y_train,X_test,y_test,fit=True)

## SMOTE ENN Performed better 
# %%

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(x_train,y_train)
evaluate_model(rf_clf,x_train,y_train,X_test,y_test,fit=True)
# %%
## Hyperparameter tunining 
# we are tuning three hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    "n_estimators" : [90,100,115,130],
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,20,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'max_features' : ['auto','log2']
}
random_search = RandomizedSearchCV(estimator=rf_clf,param_distributions=grid_param,cv=5,n_jobs =-1,verbose = 3)
random_search.fit(x_train,y_train)
# %%
# %%
print(random_search.best_params_)
# %%
rand_clf_tune = RandomForestClassifier(criterion= 'entropy',
 max_depth = 14,
 max_features = 'log2',
 min_samples_leaf = 1,
 min_samples_split= 4,
 n_estimators = 115,random_state=6)

rand_clf_tune.fit(x_train,y_train)
evaluate_model(rand_clf_tune,x_train,y_train,X_test,y_test,fit=True)
# %%

xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(x_train, y_train)
evaluate_model(xgb,x_train,y_train,X_test,y_test,fit=True)


# %%
param_grid={
   
    'learning_rate':[1,0.5,0.1,0.01,0.001],
    'max_depth': [3,5,10,20],
    'n_estimators':[10,50,100,200]
    
}
grid_xgb= RandomizedSearchCV(XGBClassifier(objective='binary:logistic'),param_grid, verbose=3)
grid_xgb.fit(x_train,y_train)
print(grid_xgb.best_params_)
# %%
xgb_tuned = XGBClassifier(objective='binary:logistic',learning_rate =0.5,max_depth = 10,n_estimators = 200)
xgb_tuned.fit(x_train, y_train)
evaluate_model(xgb_tuned,x_train,y_train,X_test,y_test,fit=True)

# %%
catboost = CatBoostClassifier()
catboost.fit(x_train,y_train)
evaluate_model(catboost,x_train,y_train,X_test,y_test,fit=True)
# %%
cbc = CatBoostClassifier()

#create the grid
grid = {'max_depth': [3,4,5,6,7,8,9],'n_estimators':[100, 200, 300]}

#instantiate  GridSearchCV
gscv = GridSearchCV (estimator = cbc, param_grid = grid, scoring = "roc_auc_ovr"
, cv = 5)

#fit the model using grid search
gscv.fit(x_train,y_train)

#returns the estimator with the best performance
print(gscv.best_estimator_)

#returns the best score
print(gscv.best_score_)

#returns the best parameters
print(gscv.best_params_)


# %%
catboost_tuned = CatBoostClassifier(max_depth = 9,n_estimators=200)
catboost_tuned.fit(x_train,y_train)
evaluate_model(catboost_tuned,x_train,y_train,X_test,y_test,fit=True)


# %%
lgbm = lgb.LGBMClassifier()
lgbm.fit(x_train, y_train)
evaluate_model(lgbm,x_train,y_train,X_test,y_test,fit=True)
# %%

param_test ={'num_leaves': sp_randint(6, 50), 
             
               "n_estimators" : [50,100,200,300]}

lgb_clf = lgb.LGBMClassifier(max_depth=7, random_state=314, silent=True, metric='None', n_jobs=4)
lgb_rs = RandomizedSearchCV(
    estimator=lgb_clf, param_distributions=param_test, 
    n_iter=100,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

lgb_rs.fit(x_train,y_train)

print('Best score reached: {} with params: {} '.format(lgb_rs.best_score_, lgb_rs.best_params_))
# %%
tunned_lgb=lgb.LGBMClassifier(n_estimators = 300,num_leaves = 44)
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100}

tunned_lgb.fit(x_train,y_train,**fit_params)
evaluate_model(tunned_lgb,x_train,y_train,X_test,y_test,fit=True)
# %%
shap.initjs()

explainer = shap.TreeExplainer(catboost_tuned)

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test,)



# %%
## Predicting for Vishesh

catboost_tuned.predict_proba([814,27,5,10000,3,1,1,90000,0,0,1])


# %%
def efficient_cutoff(actual_value,predicted):
    '''
    Model probablity threshold cutoff plot
    :param  actual_value :  Actual  target values
    :param predicted: Predicted probabilities from the model
 

    :return: Train and Test Classification report and AUC- ROC Graph
    '''
    probability_cutoff = []
    accuracy_score_val = []
    recall_score_val=[]
    for i in range(30,50,2): ## Trying different probablity threshold values
        predicted_x = deepcopy(predicted)
        predicted_x[predicted_x >= i / 100] = 1 ## Classifying class 1 as greater than threshold
        predicted_x[predicted_x < i / 100] = 0 ## Classifying class 0 as less then threshold
        probability_cutoff.append(i/100)
        accuracy_score_val.append(accuracy_score(actual_value,predicted_x)) ## Calulating Accuracy Scores
        recall_score_val.append(recall_score(actual_value,predicted_x)) ##  Caluclating Recall Scores
        
    
    return (probability_cutoff,accuracy_score_val,recall_score_val)




# %%
pred= catboost_tuned.predict_proba(X_test)
efficient_cutoff(y_test,pred[:,1])
probability_cutoff,accuracy_score_val,recall_score_val=efficient_cutoff(y_test,pred[:,1])
    
fig = px.scatter( x=accuracy_score_val, y=recall_score_val,text=probability_cutoff, title='Threshold cutoff plot', labels={
                     "y":"Recall ",
                    "x": "Accuracy",
                     },)
fig.show()

## We can observe that 0.38 could be good threshold value to improve model further
# %%
