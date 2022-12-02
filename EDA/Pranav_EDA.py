# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly


# %%
## Loading the dataset 
df=pd.read_csv(r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Data\Bank_data.csv")
df.head()
df.tail()

# %%
## Shape of the data
df.shape

# %%
## Checking column types 
df.info()
# %%
## Checking null values 
df.isnull().sum()
# %%
### Univariate Analysis 
