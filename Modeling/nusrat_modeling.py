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




#%%
#Logit Regression
#Logistic regression is to predict a binary outcome a data set


# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
modelCreditScore = glm(formula='CreditScore ~ Age+C(Tenure)+C(Balance)+C(Balance)+C(NumOfProducts)+C(HasCrCard)+C(IsActiveMember)+C(EstimatedSalary)+C(Exited)', data=df, family=sm.families.Binomial()).fit()
print(modelCreditScore.summary())
modelpredicitons = pd.DataFrame( columns=['CreditScore_df'], data= modelCreditScore.predict(df)) 
print(modelpredicitons.head())

# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
modelCreditScore = glm(formula='CreditScore ~ Age+C(Tenure)+C(Balance)', data=df, family=sm.families.Binomial()).fit()
print(modelCreditScore.summary())
modelpredicitons = pd.DataFrame( columns=['CreditScore_df'], data= modelCreditScore.predict(df)) 
print(modelpredicitons.head())
# %%
