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

x_test=pk.load(open(r"/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Preprocessed_data/x_test.pk",'rb'))
x_train=pk.load(open(r"/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Preprocessed_data/x_train.pk",'rb'))
y_test=pk.load(open(r"/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Preprocessed_data/y_test.pk",'rb'))
y_train=pk.load(open(r"/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Preprocessed_data/y_train.pk",'rb'))

print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(x_test.shape)
print(y_test.shape)

#%%
#Logit Regression



# %%
sns.heatmap(df.corr(),annot=True)
plt.show()
modelCreditScore = glm(formula='CreditScore ~ Age+C(Tenure)+C(Balance)', data=df, family=sm.families.Binomial()).fit()
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
