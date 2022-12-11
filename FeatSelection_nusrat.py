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

# Let check wheather gender has an effet on Churn or not

from scipy import stats
crosstable1=pd.crosstab(df['Gender'], df['Exited'])
print(crosstable1)
stats.chi2_contingency(crosstable1)

#Finding: Here p is less than o.o5. That accept null hypothesis.
# It can be said that Gender has no relation with the churn rate.

# Let check wheather Geography has an effet on CreditScore or not
crosstable2=pd.crosstab(df['Geography'], df['Exited'])
print(crosstable2)
stats.chi2_contingency(crosstable2)

#Finding: Here p is less than o.o5. That accept null hypothesis.
# It can be said that Geography has no effect on Churn rate.

#Let's check active member depends on gender or not

crosstable3=pd.crosstab(df['IsActiveMember'], df['Gender'])
print(crosstable3)
stats.chi2_contingency(crosstable3)

#Finding: Here p is 0.0254 which is less than o.o5. That accept the null hypothesis.
#Acrive member is not depend on gender.

#Let's check active member depends on Region or not

crosstable4=pd.crosstab(df['IsActiveMember'], df['Geography'])
print(crosstable4)
stats.chi2_contingency(crosstable4)

#Finding: Here p is 0.070 which is grater than o.o5. That reject the null hypothesis.
#Acrive member depends on Region.

#Let's check  Card Holder depends on gender or not

crosstable5=pd.crosstab(df['HasCrCard'], df['Gender'])
print(crosstable5)
stats.chi2_contingency(crosstable5)

#Finding: Here p is 0.579 which is grater than o.o5. That reject the null hypothesis.
#Card Holder depends on gender.

#Let's check Card Holder depends on Region or not

crosstable6=pd.crosstab(df['HasCrCard'], df['Geography'])
print(crosstable6)
stats.chi2_contingency(crosstable6)

#Finding: Here p is 0.327 which is grater than o.o5. That reject the null hypothesis.
#Card Holder depends on region.

#Let's check  NumofProducts depends on Geography or not

crosstable7=pd.crosstab(df['NumOfProducts'], df['Geography'])
print(crosstable7)
stats.chi2_contingency(crosstable7)
#Finding: Here p  is less than o.o5. That accept the null hypothesis.
#NumOfProducts has no relation with the geography

# %%
