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
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import rfit

#%%
df = pd.read_csv(r'/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Data/Bank_data.csv')
print(df)
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.head()
df.tail()
df.isnull().sum().to_frame(' Nulls values ')

# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
modelCreditScore = glm(formula='CreditScore ~ Age+C(Tenure)+C(Balance)', data=df, family=sm.families.Binomial()).fit()
print(modelCreditScore.summary())
modelpredicitons = pd.DataFrame( columns=['CreditScore_df'], data= modelCreditScore.predict(df)) 
print(modelpredicitons.head())
# %%
