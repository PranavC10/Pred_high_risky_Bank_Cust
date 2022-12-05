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

# %%
