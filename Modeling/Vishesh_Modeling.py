# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# %%
# Imorting dataset
Bank_data = pd.read_csv(r"C:\Users\vishesh\Documents\GitHub\T1-Data_Ninjas-22FA\Data\Bank_data.csv")

print(f"Bank_data.shape = {Bank_data.shape}")
print(f"Bank_data.info() = {Bank_data.info()}")
print(f"Bank_data.head() = {Bank_data.head()}")


# %%
## Loading preprocessed data using pickle

#x_train=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\x_train.pk", "rb" ) )
#y_train=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\y_train.pk", "rb" ) )

#x_test=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\x_test.pk", "rb" ) )
#y_test=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\y_test.pk", "rb" ) )
#print("Train Data Shape")
#print(x_train.shape)
#print(y_train.shape)

#print("Test Data Shape")
#print(x_test.shape)
#print(y_test.shape)

# %%
# Creating Test Train split
x=Bank_data[["CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance","NumOfProducts","HasCrCard", "IsActiveMember", "EstimatedSalary"]]
y=Bank_data["Exited"]
							

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=321)

# %%

clf = GaussianNB()
clf.fit(x_train, y_train)

# %%
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
# %%

y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))

# %%
