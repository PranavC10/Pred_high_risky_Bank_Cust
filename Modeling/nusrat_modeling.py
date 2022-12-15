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




# %%

# Logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

creditlogit = LogisticRegression() 
creditlogit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', creditlogit.score(x_test, y_test))
print('Logit model accuracy (with the train set):', creditlogit.score(x_train, y_train))

print("\nReady to continue.")


y_pred=creditlogit.predict(x_test)
print(classification_report(y_test,y_pred))
print("\nReady to continue.")

#%%



#%%


#%%


# %%
