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

plt.hist([churnF,notchurnF], label='age',edgecolor='black', linewidth=1.2,alpha = 0.7)
plt.hist([churnG,notchurnG], label='age',edgecolor='black', linewidth=1.2,alpha = 0.7)
plt.hist([churnS,notchurnS], label='age',edgecolor='black', linewidth=1.2,alpha = 0.7)
plt.xlabel('Age of Both Male & Female')
plt.ylabel('Number of People')
plt.savefig('hist_educ.png')
plt.show()

print('\n','French has the highest number of customers churned.','\n')
# %%
