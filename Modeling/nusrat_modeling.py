#%%
#import libaries
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import pickle as pk
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import rfit

#%%
df = pd.read_csv(r'/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Data/Bank_data.csv')
print(df)
df = df.drop(["RowNumber","CustomerId"],axis=1)
df.head()
df.tail()
print('Data finding')
print(type(df))
print(df.dtypes)
df.describe()

print('finding mising value')
print(df.isnull().sum())




# %%
#Corr plot
sns.heatmap(df.corr(),annot=True)
plt.show()


# %%
#Spliting
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


xdf = df[['Age', 'Tenure', 'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']]
ydf = df['CreditScore']
print(type(xdf))
print(type(ydf))


#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdf, ydf,test_size=0.25, random_state=1)

print('x_train type',type(x_train))
print('x_train shape',x_train.shape)
print('x_test type',type(x_test))
print('x_test shape',x_test.shape)
print('y_train type',type(y_train))
print('y_train shape',y_train.shape)
print('y_test type',type(y_test))
print('y_test shape',y_test.shape)

print("\nReady to continue.")
# %%

# Logit
from sklearn.linear_model import LogisticRegression

creditlogit = LogisticRegression()  # instantiate
creditlogit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', creditlogit.score(x_test, y_test))
print('Logit model accuracy (with the train set):', creditlogit.score(x_train, y_train))

print("\nReady to continue.")

#%%
print(creditlogit.predict(x_test))

print("\nReady to continue.")
print(creditlogit.predict_proba(x_train[:8]))
print(creditlogit.predict_proba(x_test[:8]))

print("\nReady to continue.")

#%%


#%%


# %%
