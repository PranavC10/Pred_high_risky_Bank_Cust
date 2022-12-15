
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
df = df.drop(["RowNumber","CustomerId","Surname"],axis=1)
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
#CreditScore
sns.distplot(df.CreditScore)
plt.ylabel('Frequency', fontsize=15)
plt.xlabel('Credit Score', fontsize=15)
plt.title('Credit Score Frequency Distribution', fontsize=15)
plt.show()
#Most of the distributions are between 600 and 700. And the distribution is normal.
#%%
#Age
np.sort(df.Age.unique())
#We have people of age between 18 and 92 inclusive.
fig = px.box(df, x="Age")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
# Mostly age group people are in the dataset is 32-44
fig = px.box(df, x="Exited",y="Age")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
# Age between 38-52 effect the churn rate
#CreditScore with Age
plt.figure(figsize=(15, 8))
sns.scatterplot(x='CreditScore', data=df, y='Age')
#No corelation fould with age and credit Score

#%%
print('Unique years of tenures:', list(np.sort(df.Tenure.unique())))
print(df['Tenure'].mean())
sns.countplot(x=df.Tenure)
plt.title('Count of each tenure', fontsize=15)
plt.xlabel('Tenure', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()
# people have tenures of 0 to 10 years. Mostly all people have same year of tenures.

df[['Tenure', 'Exited']].corr()
#Poor corelation found here. Tenure doesn't matter when it comes to churn rate
#%%
#Numeber of Product
print(df.NumOfProducts.unique())
plt.figure(figsize=(15, 8))
sns.countplot(x=df.NumOfProducts)
plt.show()
#customers have at least 1 product and at most 4 products.Most customers own 1 or 2 products.
# %%
#Estimated salary
print(df['EstimatedSalary'].mean())
sns.distplot(df.EstimatedSalary)
plt.show()
#Average Salary of customer is 1lace. Most people has the same amount of salary range
# %%
df.groupby("Gender").mean()["CreditScore"].plot(kind='bar')
plt.title("Gender interates CreditScore ")
plt.show()

# %%



# %%
