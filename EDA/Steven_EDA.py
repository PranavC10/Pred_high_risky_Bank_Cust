# %%
# Import package
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
# %%
#Load Data
df=pd.read_csv(r"/Users/stephantian/Desktop/Grad School/Course/6103 Data Mining/T1-Data_Ninjas-22FA/Data/Bank_data.csv")
df.head()
df.tail()
# %%
geo = df['Geography']
churnF = df[df['Geography'] == 'France']
churnG = df[df['Geography'] == 'Germany']
churnS = df[df['Geography'] == 'Spain']

x=df['Geography'].value_counts()
x=x.sort_index()
#plot
churn = df['Geography']
churn.value_counts().plot(kind='pie')
plt.xlabel('# of Churn by country', fontsize=12)
# %%
