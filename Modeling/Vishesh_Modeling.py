# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from statsmodels.formula.api import ols

# %%
# Imorting dataset
Bank_data = pd.read_csv(r"C:\Users\vishesh\Documents\GitHub\T1-Data_Ninjas-22FA\Data\Bank_data.csv")

print(f"Bank_data.shape = {Bank_data.shape}")
print(f"Bank_data.info() = {Bank_data.info()}")
print(f"Bank_data.head() = {Bank_data.head()}")

# %%

## Dropping unwanted columns like RowNumber,CustomerID,surname
## This columns won't help in prediction of target 
Bank_data.drop(columns=["CustomerId","Surname"],inplace=True)
Bank_data.head(1)

# %%

## Converting Catgorical columns to numerical data using one hot encoding 
##  Geography and Gender both are nominal data hence one hot encoding would be better option
cat_col=["Geography","Gender"]
Bank_data= pd.get_dummies(Bank_data, columns=cat_col, drop_first=True)
Bank_data.head(1)

# %%

## Spliting dataset into Training and Testing 
x=Bank_data[["RowNumber", "CreditScore", "Geography_Germany", "Geography_Spain", "Gender_Male", "Age", "Tenure", "Balance","NumOfProducts","HasCrCard", "IsActiveMember", "EstimatedSalary"]]
y=Bank_data["Exited"]

x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.25,random_state=1)

# %%


clf = GaussianNB()
clf.fit(x_train, y_train)


print("Train score",clf.score(x_train,y_train))
print("Test score",clf.score(x_test,y_test))

y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))

# %%

## Using SMOTE To balance Training dataset 
# transform the dataset
oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

y_train.value_counts()


# %%

#merge_data = pd.merge(x_train,y_train, on=["ID"])

model_b = ols(formula="Exited ~ CreditScore + Geography_Germany + Geography_Spain + Gender_Male + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary", data=Bank_data)
print(type(model_b))

model_bFit = model_b.fit()
print(type(model_bFit))
print(model_bFit.summary())

# %%

## Spliting dataset into Training and Testing 
x=Bank_data[["CreditScore", "Gender_Male", "Age", "Balance","NumOfProducts","HasCrCard", "IsActiveMember", "EstimatedSalary"]]
y=Bank_data["Exited"]

x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.25,random_state=1)
# %%

## Using SMOTE To balance Training dataset 
# transform the dataset
#oversample = SMOTE()
#x_train, y_train = oversample.fit_resample(x_train, y_train)

#y_train.value_counts()
# %%

clf = GaussianNB()
clf.fit(x_train, y_train)


print("Train score",clf.score(x_train,y_train))
print("Test score",clf.score(x_test,y_test))

y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))

# %%
