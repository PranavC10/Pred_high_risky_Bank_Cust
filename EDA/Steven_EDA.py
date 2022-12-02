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
churnF = df[df['Geography'] == 'France']
churnG = df[df['Geography'] == 'Germany']
churnS = df[df['Geography'] == 'Spain']

# General description of countries
df['Geography'].describe()

#Pie chart of country country (Germany, Spain, France) with highest churn rate
churn = df['Geography']
churn.value_counts().plot(kind='pie')
plt.xlabel('# of Churn by country', fontsize=12)
plt.show()

print('\n','French has the highest number of customers churned.','\n')
# %%
