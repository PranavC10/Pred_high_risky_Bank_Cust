# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px


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
## Plotting pie char to understand % of customer exited 
labels = ["Current Customers","Exited Customers"]
values = list(df["Exited"].value_counts()[0:2])

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels,textinfo='label+percent', values=values, pull=[0, 0.2])])
fig.show()


# %%

fig = px.box(df, x="Exited", y="CreditScore")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

# %%
fig = px.box(df, x="Exited", y="Balance")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

# %%
fig = px.box(df, x="Exited", y="EstimatedSalary")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
# %%
