# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# %%
## Loading preprocessed data (train and test split) from pickle fiile


x_train=pk.load( open( r"C:\Users\vishe\OneDrive\Documents\GitHub\T1-Data_Ninjas-22FA\Preprocessed_data\x_train.pk", "rb" ) )
y_train=pk.load( open( r"C:\Users\vishe\OneDrive\Documents\GitHub\T1-Data_Ninjas-22FA\Preprocessed_data\y_train.pk", "rb" ) )

x_test=pk.load( open( r"C:\Users\vishe\OneDrive\Documents\GitHub\T1-Data_Ninjas-22FA\Preprocessed_data\x_test.pk", "rb" ) )
y_test=pk.load( open( r"C:\Users\vishe\OneDrive\Documents\GitHub\T1-Data_Ninjas-22FA\Preprocessed_data\y_test.pk", "rb" ) )


#Printing data shape
print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(x_test.shape)
print(y_test.shape)

# %%

#Naive Bayes model
clf = GaussianNB()
clf.fit(x_train, y_train)

#Naive Bayes model Test scores
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))


#Naive Bayes model AUC ROC graph
probas = clf.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

plt.figure(figsize=((5,2)))
plt.plot([0,1],[0,1],'k--') #plot the diagonal line
plt.plot(fpr, tpr, label='NB') #plot the ROC curve
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Naive Bayes')
plt.show()

print(roc_auc_score(y_test, probas[:, 1]))

# %%
