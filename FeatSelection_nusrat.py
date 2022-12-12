#%%
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

# %%
#Chi-Square Test
#Goodness of fit
#Gender
import scipy.stats as stats
Observed1= list(df['Gender'].value_counts())
Expected1= [sum(Observed1)/len(Observed1)]

#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed1, f_exp=Expected1)

# %%

