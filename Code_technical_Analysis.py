# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.linear_model  import  LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import  roc_curve, roc_auc_score
import plotly.express as px
import matplotlib.pyplot as plt
import pickle as pk
from copy import deepcopy
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

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






# %%
# Separate each country from Dataset
geo = df['Geography']
F = df[df['Geography'] == 'France']
G = df[df['Geography'] == 'Germany']
S = df[df['Geography'] == 'Spain']

# General description of countries
df['Geography'].describe()

#Pie chart of country country (Germany, Spain, France) with highest churn rate
churn = df['Geography']
churn.value_counts().plot(kind='pie')
plt.xlabel('# of Churn by country', fontsize=12)
plt.savefig('pie_country_distribution')
plt.show()

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
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)
X_train, X_test, Y_train, y_test = train_test_split(X_scaled, y,stratify=y, test_size=0.25,random_state=1)


# %%
## Logistic Regression 
log_reg = LogisticRegression()

log_reg.fit(X_train,Y_train)
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


def evaluate_model(model,x_train,y_train,x_test,y_test,fit=False,threshold_graph = False):
    
    if fit== False:
        model.fit(x_train,y_train)
        
    
    train_pred=model.predict(x_train)
    print("Training report")
    print(accuracy_score(y_train,train_pred))
    print(classification_report(y_train, train_pred))
    
    print("Testing report")
    test_pred=model.predict(x_test)
    print("Accuracy")
    
    print("F1_Score")
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
    
    if threshold_graph == True:
        
        probability_cutoff,accuracy_score_val,recall_score_val=efficient_cutoff(y_test,y_pred_prob[:,1])
    
        fig = px.scatter( x=accuracy_score_val, y=recall_score_val,text=probability_cutoff, title='Threshold cutoff plot')
        fig.show()


    
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
print(y_train.values_count())
evaluate_model(log_reg_smote,x_train,y_train,X_test,y_test,fit=True)


# %%
smt = SMOTEENN(random_state=42)
x_train, y_train = smt.fit_resample(X_train, Y_train)

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
rand0m_search = RandomizedSearchCV(estimator=clf,param_distributions=grid_param,cv=5,n_jobs =-1,verbose = 3)
rand0m_search.fit(x_train,y_train)
# %%
# %%
print(rand0m_search.best_params_)
# %%
rand_clf_tune = RandomForestClassifier(criterion= 'entropy',
 max_depth = 14,
 max_features = 'log2',
 min_samples_leaf = 1,
 min_samples_split= 4,
 n_estimators = 115,random_state=6)

rand_clf_tune.fit(X_train,y_train)
evaluate_model(rand_clf_tune,X_train,y_train,X_test,y_test,fit=True)

# %%
#KNN Model
print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(X_test.shape)
print(y_test.shape)

#Import KNN package
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # instantiate with n value given
knn.fit(X_train,y_train)
y_pred = knn.predict(X_train)
y_pred = knn.predict_proba(X_train)
print(y_pred)
print(knn.score(X_train,y_train))
print('\n','Next Line')
from sklearn.neighbors import KNeighborsClassifier
knn_split = KNeighborsClassifier(n_neighbors=1)
knn_split.fit(X_train,y_train)
ytest_pred = knn_split.predict(X_test)
ytest_pred
print(knn_split.score(X_test,y_test))

from sklearn.model_selection import GridSearchCV
para = {'n_neighbors':[1,2,3,5,6,7,9,10]}
knn_cv = KNeighborsClassifier()
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv,X_train, y_train, cv=7)
n = GridSearchCV(knn_cv,para,cv=7)
n.fit(X_train,y_train)
print({(n.best_params_)['n_neighbors']})
print(n.best_score_)
print(cv_results) 
print(np.mean(cv_results)) 
# %%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
 
cm = confusion_matrix(y_test,ytest_pred)
print(cm)
 
color = 'white'
matrix = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()

# %%
from sklearn.metrics import classification_report
 
print(classification_report(y_test, ytest_pred))

