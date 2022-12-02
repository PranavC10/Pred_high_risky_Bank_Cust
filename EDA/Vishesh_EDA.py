
#%%
#imports

import pandas as pd
import plotly.express as px

#%%
# data import

Bank_data = pd.read_csv('Bank_data.csv')
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

fig = px.histogram(Bank_data, x="CreditScore", color="Exited", nbins=60, title="Impact of Credit score on customer churn rate")

fig.update_layout(
    title="Impact of Credit score on customer churn rate",
    xaxis_title="Credit Score",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

print(h1 + "Finding:" + e + s1 + "Credit score in general does not have an impact on the churn rate, but in the grpah we can see that Credit score and Churn rate are normally distributed and the maximum population have credit score between 600-700 and this population has highest churn rate as well. Also we can see that churn rate is directely proportional to population. Greater the population in the bin greatr is the churn rate" + e)
print()

# %%
#2 Balance vs churn rate

fig = px.box(Bank_data, x="Balance", color="Exited", title="Impact of Customer Balance on customer churn rate")

fig.update_layout(
    title="Impact of Customer Balance on customer churn rate",
    xaxis_title="Customer Balance",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()


fig = px.histogram(Bank_data, x="Balance", color="Exited",
                   marginal="box")
fig.show()

# %%
#3 Estimated salary vs churn rate

fig = px.box(Bank_data, x="EstimatedSalary", color="Exited", title="Impact of Customer Balance on customer churn rate")

fig.update_layout(
    title="Impact of Customer Balance on customer churn rate",
    xaxis_title="Customer Balance",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

# %%

#4 Age vs churn rate

fig = px.histogram(Bank_data, x="Age", color="Exited", title="Impact of Customer Balance on customer churn rate")

fig.update_layout(
    title="Impact of Customer Balance on customer churn rate",
    xaxis_title="Age",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

# %%

# Customer churn rate based on Gender

fig = px.histogram(Bank_data, x="Gender",color="Exited", title = "Customer churn rate based on Gender")
fig.show()

#fig = px.pie(Bank_data, values='Exited', names='Gender')
#fig.show()


# %%
