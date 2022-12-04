# %%
# Import package needed
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# %%
#Load CSV Data
df=pd.read_csv(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Data/Bank_data.csv")
df.head()
df.tail()
# %%
# Separate each country from Dataset
geo = df['Geography']
F = df[df['Geography'] == 'France']
G = df[df['Geography'] == 'Germany']
S = df[df['Geography'] == 'Spain']

# General description of countries
df['Geography'].describe()

#Pie chart of country country (Germany, Spain, France) with highest churn rate
churn = df['Geography']
churn.value_counts().plot(kind='pie')
plt.xlabel('# of Churn by country', fontsize=12)
plt.savefig('pie_country_distribution')
plt.show()

# Distribution of each country in the dataset 

# Separate each country into churn and not churn
churnF = F[F['Exited'] == 1]
notchurnF = F[F['Exited'] == 0]
churnG = G[G['Exited'] == 1]
notchurnG = G[G['Exited'] == 0]
churnS = S[S['Exited'] == 1]
notchurnS = S[S['Exited'] == 0]
#height = [1500,2000,2500,3000,3500,4000,4500,5000,5500]

plt.xticks(np.arange(0,1.1, step=1))
plt.hist([churnF['Exited'],notchurnF['Exited'],churnG['Exited'],notchurnG['Exited'],churnS['Exited'],notchurnS['Exited']], 
        label=['France churn','France not churn','Germany churn','Germany not churn','Spain churn','Spain not churn'],
        histtype = 'bar', rwidth = 1,bins=[0, 1],edgecolor = 'black')
plt.xlabel('Churn and not churn')
plt.ylabel('Number of People')
plt.legend(loc = 0)
plt.savefig('Country churn.png')
plt.show()

# Numbers of churn by country
print('Total Numbers of France customers churned',len(churnF))
print('Total Numbers of Germany customers churned',len(churnG))
print('Total Numbers of Spain customers churned',len(churnS))

# Churn rate of each country
print('France churn rate',round(len(churnF)/len(F)*100,2))
print('Germany churn rate',round(len(churnG)/len(G)*100,2))
print('Spain churn rate',round(len(churnS)/len(S)*100,2))

print('\n')
print('Germany has the highest number of customers churned among all countries with 814 customers churned.')
print('\n')
print('Germany also has the highest churn rate of 32.44%','\n')
# %%
# Data Prepocessing 