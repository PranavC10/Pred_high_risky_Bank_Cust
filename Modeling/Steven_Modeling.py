#%%
#Import package going to be used for model

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle as pk

#%%
x_test=pk.load(open(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Preprocessed_data/x_test.pk",'rb'))
x_train=pk.load(open(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Preprocessed_data/x_train.pk",'rb'))
y_test=pk.load(open(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Preprocessed_data/y_test.pk",'rb'))
y_train=pk.load(open(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Preprocessed_data/y_train.pk",'rb'))

print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(x_test.shape)
print(y_test.shape)
# %%
#KNN Model
#Import KNN package
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # instantiate with n value given
knn.fit(x_train,y_train)
y_pred = knn.predict(x_train)
y_pred = knn.predict_proba(x_train)
print(y_pred)
print(knn.score(x_train,y_train))
print('\n','Next Line')
from sklearn.neighbors import KNeighborsClassifier
knn_split = KNeighborsClassifier(n_neighbors=1)
knn_split.fit(x_train,y_train)
ytest_pred = knn_split.predict(x_test)
ytest_pred
print(knn_split.score(x_test,y_test))

# %%
from sklearn.model_selection import GridSearchCV
para = {'n_neighbors':[1,2,3,5,6,7,9,10]}
knn_cv = KNeighborsClassifier()
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv,x_train, y_train, cv=7)
n = GridSearchCV(knn_cv,para,cv=7)
n.fit(x_train,y_train)
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
matrix = plot_confusion_matrix(knn, x_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.savefig('K Plot')
plt.show()

# %%
from sklearn.metrics import classification_report
 
print(classification_report(y_test, ytest_pred))
# %%
