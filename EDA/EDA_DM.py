
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

# %%

#Bank_data['Exited'].replace(0, 'Retiained',inplace=True)
#Bank_data['Exited'].replace(1, 'Exited',inplace=True)


#1 Credit score vs churn rate

fig = px.histogram(Bank_data, x="CreditScore", color="Exited", nbins=60, title="Impact of Credit score on customer churn rate")

fig.update_layout(
    title="Impact of Credit score on customer churn rate",
    xaxis_title="Credit Score",
    yaxis_title="Count",
    legend_title="Exited"
)

fig.show()

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

# %%

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


fig = px.violin(Bank_data, x="Gender", color="Exited", box=True, points="all",
          hover_data=Bank_data.columns)
fig.show()
# %%



fig = px.pie(Bank_data, values='Exited', names='Gender')
fig.show()
# %%
Bank_data.groupby("Gender").plot.pie(y="Exited")
# %%
Bank_data[["Exited","Gender"]]
# %%
fig = px.histogram(Bank_data, x="Gender",color="Exited")
fig.show()
# %%
Bank_data["Gender"].plot.pie()
# %%
fig = px.pie(Bank_data, values='Exited')
fig.show()
# %%
