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
Observed2= list(df['Geography'].value_counts())
Expected2= [sum(Observed2)/len(Observed2)]

#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed2, f_exp=Expected2)

# %%
Observed3=list(df['NumOfProducts'].value_counts())
Expected3= [sum(Observed3)/len(Observed3)]
#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed3, f_exp=Expected3)

# %%
Observed4=list(df['IsActiveMember'].value_counts())
Expected4= [sum(Observed4)/len(Observed4)]
#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed4, f_exp=Expected4)
# %%
Observed5=list(df['HasCrCard'].value_counts())
Expected5= [sum(Observed5)/len(Observed5)]
#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed5, f_exp=Expected5)

#%%
Observed6=list(df['Exited'].value_counts())
Expected6= [sum(Observed6)/len(Observed6)]
#perform Chi-Square Goodness of Fit Test
stats.chisquare(f_obs=Observed6, f_exp=Expected6)

# %%
