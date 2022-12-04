# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk
from sklearn.metrics import classification_report


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
