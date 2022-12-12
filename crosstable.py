#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 


#%%
df = pd.read_csv(r'/Users/nusratprithee/Documents/T1-Data_Ninjas-22FA/Data/Bank_data.csv')
print(df)
df.head()
df.tail()


crosstable1=pd.crosstab(df['Gender'], df['Exited'])
print(crosstable1)


crosstable2=pd.crosstab(df['Geography'], df['Exited'])
print(crosstable2)


crosstable3=pd.crosstab(df['IsActiveMember'], df['Gender'])
print(crosstable3)


crosstable4=pd.crosstab(df['IsActiveMember'], df['Geography'])
print(crosstable4)


crosstable5=pd.crosstab(df['HasCrCard'], df['Gender'])
print(crosstable5)


crosstable6=pd.crosstab(df['HasCrCard'], df['Geography'])
print(crosstable6)


crosstable7=pd.crosstab(df['NumOfProducts'], df['Geography'])
print(crosstable7)

