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

#%%
#Test of Independence
# chi Squre test

# Let check wheather gender has an effet on CreditScore or not

from scipy import stats
crosstable1=pd.crosstab(df['Gender'], df['CreditScore'])
print(crosstable1)
stats.chi2_contingency(crosstable1)

#Finding: Here p is 0.379 which is grater than o.o5. That reject null hypothesis.
# It can be said that CreditScore has a relation with the varialble gender.

# Let check wheather Geography has an effet on CreditScore or not
crosstable2=pd.crosstab(df['Geography'], df['CreditScore'])
print(crosstable2)
stats.chi2_contingency(crosstable2)

#Finding: Here p is 0.0874 which is grater than o.o5. That reject null hypothesis.
# It can be said that Geography has an effect on CreditScore.

# Let check wheather Card Balance depends on Gender
crosstable3=pd.crosstab(df['Gender'], df['Balance'])
print(crosstable3)
stats.chi2_contingency(crosstable3)

#Finding: Here p is 0.511 which is grater than o.o5. That reject null hypothesis.
# It can be said that Card Balance depends on Gender

