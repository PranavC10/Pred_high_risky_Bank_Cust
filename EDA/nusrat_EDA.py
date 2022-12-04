# Do customers with low credit scores who have credit cards affect the churn rate?
#%%

#import libaries
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import rfit

#%%

df = pd.read_csv(r'/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Data/Bank_data.csv')
print(df)
df.head()
df.tail()
# %%
print('Data finding')
print(type(df))
print(df.dtypes)
df.describe()

print('finding mising value')
print(df.isnull().sum())

# %%
creditscore_count = df['CreditScore'].value_counts()
print(creditscore_count)

sns.boxplot(data=df, x="CreditScore")

# %%
print('Histogram of CreditScore  by CreditCard')
df.hist(column='CreditScore', by='HasCrCard', grid=True, xlabelsize=None, xrot=None,
            ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False,
            figsize=None, layout=None, bins=30, backend=None, legend=True)
# %%
#Test of Independence
# chi Squre test
# Let check wheather gender has an effet on CreditScore or not

from scipy import stats
# World1
crosstable1=pd.crosstab(df['Gender'], df['CreditScore'])
print(crosstable1)
stats.chi2_contingency(crosstable1)

#Finding: Here p is 0.379 which is grater than o.o5. That reject null hypothesis.
# It can be said that CreditScore has a relation with the varialble gender.

# %%



# %%
