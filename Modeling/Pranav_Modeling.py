# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
# %%
## Loading preprocessed data using pickle

x_train=pk.load( open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_train.pk", "rb" ) )
y_train=pk.load( open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_train.pk", "rb" ) )

x_test=pk.load( open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_test.pk", "rb" ) )
y_test=pk.load( open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_test.pk", "rb" ) )
print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(x_test.shape)
print(y_test.shape)

# %%

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=15,min_samples_split=50,random_state=40,criterion="entropy",n_estimators=350,n_jobs=-1)
clf.fit(x_train, y_train)

# %%
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
# %%
y_test_pred = clf.predict(x_test)
print(classification_report(y_test, y_test_pred))

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
print(rand0m_search.best_params_)
# %%
rand_clf = RandomForestClassifier(criterion= 'gini',
 max_depth = 15,
 max_features = 'auto',
 min_samples_leaf = 1,
 min_samples_split= 3,
 n_estimators = 115,random_state=6)

rand_clf.fit(x_train,y_train)
# %%
print(rand_clf.score(x_train,y_train))
print(rand_clf.score(x_test,y_test))
y_test_pred = rand_clf.predict(x_test)
print(classification_report(y_test, y_test_pred))
# %%

## SVM
SVM=SVC()
SVM.fit(x_train,y_train)
# %%
print(SVM.score(x_train,y_train))
print(SVM.score(x_test,y_test))
y_test_pred = SVM.predict(x_test)

print(classification_report(y_test, y_test_pred))

# %%
## Hyperparameter tuning 
param_grid={'C':[0.1,1,0.2,0.5],'gamma':[1,0.5,00.001]} #Gamma is a scalar that defines how much influence a single training example (point) has.
grid= GridSearchCV(SVC(),param_grid, verbose=3, n_jobs=-1)
grid.fit(x_train,y_train)
print(grid.best_params_)
# %%
SVM_tuned=SVC(C=1, gamma=0.001)
SVM_tuned.fit(x_train,y_train)
print(SVM_tuned.score(x_train,y_train))
print(SVM_tuned.score(x_test,y_test))
y_test_pred = SVM_tuned.predict(x_test)

print(classification_report(y_test, y_test_pred))
# %%
