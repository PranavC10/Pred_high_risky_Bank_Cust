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



#%%
#Logit Regression
#Logistic regression is to predict a binary outcome a data set
LR_model = glm(Exited ~ ., data = df, family = "binomial")
summary(LR_model)

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
