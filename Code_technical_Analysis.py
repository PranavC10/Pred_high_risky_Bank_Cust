# %%
# Imorting all the necessary libraries 

import pandas as pd 
import numpy as np
import seaborn as sns 
import plotly
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.linear_model  import  LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,roc_curve, roc_auc_score

import plotly.express as px
import matplotlib.pyplot as plt
import pickle as pk
from copy import deepcopy
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import shap

import warnings
warnings.filterwarnings("ignore")
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
# %%

fig = px.box(df, x="Exited", y="CreditScore")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

# %%
fig = px.box(df, x="Exited", y="Balance")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

# %%
fig = px.box(df, x="Exited", y="EstimatedSalary")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
# %%

### Place holder for EDA from all 











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



def evaluate_model(model,x_train,y_train,x_test,y_test,fit=False,threshold_graph = False):
    
    if fit== False:
        model.fit(x_train,y_train)
        
    
    train_pred=model.predict(x_train)
    print("Training report")
    print(accuracy_score(y_train,train_pred))
    print(classification_report(y_train, train_pred))
    
    print("Testing report")
    test_pred=model.predict(x_test)    
    print(f1_score(y_test,test_pred))
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
log_reg.coef_
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

#Instantiate GridSearchCV
gscv = GridSearchCV (estimator = cbc, param_grid = grid, scoring = "roc_auc_ovr"
, cv = 5)

#fit the model
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
    probability_cutoff = []
    accuracy_score_val = []
    recall_score_val=[]
    for i in range(30,50,2):
        predicted_x = deepcopy(predicted)
        predicted_x[predicted_x >= i / 100] = 1
        predicted_x[predicted_x < i / 100] = 0
        probability_cutoff.append(i/100)
        accuracy_score_val.append(accuracy_score(actual_value,predicted_x))
        recall_score_val.append(recall_score(actual_value,predicted_x))
        
    
    return (probability_cutoff,accuracy_score_val,recall_score_val)




# %%
pred= catboost_tuned.predict_proba(X_test)
efficient_cutoff(y_test,pred[:,1])
probability_cutoff,accuracy_score_val,recall_score_val=efficient_cutoff(y_test,pred[:,1])
    
fig = px.scatter( x=accuracy_score_val, y=recall_score_val,text=probability_cutoff, title='Threshold cutoff plot')
fig.show()