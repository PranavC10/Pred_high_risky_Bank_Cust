# %%
# Imorting all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pk
from imblearn.over_sampling import SMOTE

# %%
# Loading the dataset 
df=pd.read_csv(r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Data\Bank_data.csv")
df.head()
df.tail()
df["Exited"].value_counts( normalize = True)
df["Exited"].value_counts( )


# %%

## Dropping unwanted columns like RowNumber,CustomerID,surname
## This columns won't help in prediction of target 
df.drop(columns=["RowNumber","CustomerId","Surname"],inplace=True)
df.head(1)
# %%
## Converting Catgorical columns to numerical data using one hot encoding 
##  Geography and Gender both are nominal data hence one hot encoding would be better option
cat_col=["Geography","Gender"]
df= pd.get_dummies(df, columns=cat_col, drop_first=True)
# %%
## Spliting dataset into Training and Testing 
x=df.drop(columns=["Exited"])
y=df["Exited"]


X_train, X_test, Y_train, y_test = train_test_split(x, y,stratify=y, test_size=0.25,random_state=1)
# %%

## Using SMOTE To balance Training dataset 
# transform the dataset

#oversample = SMOTE()
#x_train, y_train = oversample.fit_resample(X_train, y_train)

#y_train.value_counts()
# %%
## Saving the data using pickle for fast access 

#pk.dump( x_train, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_train.pk", "wb" ) )
#pk.dump( y_train, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_train.pk", "wb" ) )

#pk.dump( X_test, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_test.pk", "wb" ) )
#pk.dump( y_test, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_test.pk", "wb" ) )


# %%
### Undersampling 
### Random Under-Sampling
### Undersampling can be defined as removing some observations of the majority class. This is done until the majority and minority class is balanced out.

print(df["Exited"].value_counts())

# %%
## Oversampling 
# class_count_0, class_count_1 = df["Exited"].value_counts()



#class_0 = df[df['Exited'] == 0]
#class_1 = df[df['Exited']  == 1]
# print the shape of the class
#print('class 0:', class_0.shape)
#print('class 1:', class_1.shape)


#class_1_over = class_1.sample(class_count_0, replace=True)

#test_over = pd.concat([class_1_over, class_0], axis=0)

#print("total class of 1 and 0:",test_over['Exited'].value_counts())# plot the count after under-sampeling
#test_over['Exited'].value_counts().plot(kind='bar', title='count (target)')



# %%
from imblearn.combine import SMOTEENN
smt = SMOTEENN(random_state=42)
x_train, y_train = smt.fit_resample(X_train, Y_train)


pk.dump( x_train, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_train.pk", "wb" ) )
pk.dump( y_train, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_train.pk", "wb" ) )

pk.dump( X_test, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\x_test.pk", "wb" ) )
pk.dump( y_test, open( r"C:\Users\Pranav\My Projects\T1-Data_Ninjas-22FA\Preprocessed_data\y_test.pk", "wb" ) )


# %%
