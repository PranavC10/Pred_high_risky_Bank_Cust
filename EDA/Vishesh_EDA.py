
#%%
#imports

import pandas as pd
import plotly.express as px

#%%
# data import

Bank_data = pd.read_csv(r"C:\Users\vishesh\Documents\GitHub\T1-Data_Ninjas-22FA\Data\Bank_data.csv")
print(f"Bank_data.shape = {Bank_data.shape}")
print(f"Bank_data.info() = {Bank_data.info()}")
print(f"Bank_data.head() = {Bank_data.head()}")


# print styling
s1 = '\033[1m' + '\033[96m'
s2 = '\033[1m' + '\033[4m' + '\033[95m'
h1 = '\033[1m' + '\033[4m' + '\033[92m'
e = "\033[0;0m"

# %%

#1 Credit score vs churn rate

fig = px.histogram(Bank_data, x="CreditScore", color="Exited", color_discrete_map={"Exited":"#ffc8dd"}, nbins=60, title="Impact of Credit score on customer churn rate")

fig.update_layout(
    title="Impact of Credit score on customer churn rate",
    xaxis_title="Credit Score",
    yaxis_title="Count",
    legend_title="Exited",
)

fig.show()

print(h1 + "Finding:" + e + s1 + "Credit score in general does not have an impact on the churn rate, but in the grpah we can see that Credit score and Churn rate are normally distributed and the maximum population have credit score between 600-700 and this population has highest churn rate as well. Also we can see that churn rate is directely proportional to population. Greater the population in the bin greatr is the churn rate" + e)

#color_discrete_sequence=px.colors.qualitative.Antique,

# %%
#2 Balance vs churn rate

fig = px.box(Bank_data, x="Balance", color="Exited", color_discrete_map={"Exited":"#ffc8dd"}, title="Impact of Balance on churn rate")

fig.update_layout(
    title="Impact of Customer Balance on churn rate",
    xaxis_title="Customer Balance",
    yaxis_title="Count",
    legend_title="Exited"
    
)

fig.show()


#fig = px.histogram(Bank_data, x="Balance", color="Exited",
#                   marginal="box")
#fig.show()

print(h1 + "Finding:" + e + s1 + "From the graph we can see that people with higher balance tend to churn more than the one with the low balance." + e)

# %%
#3 Estimated salary vs churn rate

fig = px.box(Bank_data, x="EstimatedSalary", color="Exited", color_discrete_map={"Exited":"#ffc8dd"})

fig.update_layout(
    title="Impact of Salary on churn rate",
    xaxis_title="Estimated Salary",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

print(h1 + "Finding:" + e + s1 + "From the graph we can see that Estimated Salary does not have much effect on the customer churn rate the difference is very minute. " + e)


# %%

#4 Age vs churn rate

fig = px.histogram(Bank_data, x="Age", color="Exited", color_discrete_map={"Exited":"#ffc8dd"})

fig.update_layout(
    title="Impact of Age on customer churn rate",
    xaxis_title="Age",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

print(h1 + "Finding:" + e + s1 + "The graph is right skewed for the reatined customer, but the graph looks slightly normal for the exited customers. It can be seen that customer tend to churn more between the age of 40-60." + e)

# %%

# Customer churn rate based on Gender

fig = px.histogram(Bank_data, x="Gender",color="Exited", color_discrete_map={"Exited":"#ffc8dd"})
fig.show()

fig.update_layout(
    title="Customer churn rate based on Gender",
    xaxis_title="Age",
    yaxis_title="Count",
    legend_title="Exited",
    text= "counts"
)

print(h1 + "Finding:" + e + s1 + "Even though the count of Male customer is more than the Female customer, the churn couunt of female customer is more than the male customer " + e)

# %%
