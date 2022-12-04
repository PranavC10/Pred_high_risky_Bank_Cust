# %%
# Imorting all the necessary libraries 

import pandas as pd
import numpy as np
import pickle as pk


# %%
## Loading preprocessed data using pickle

x_train=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\x_train.pk", "rb" ) )
y_train=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\y_train.pk", "rb" ) )

x_test=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\x_test.pk", "rb" ) )
y_test=pk.load( open( r"C:\Users\vishesh\Documents\GWU\Data Mining\Project\Program\y_test.pk", "rb" ) )
print("Train Data Shape")
print(x_train.shape)
print(y_train.shape)

print("Test Data Shape")
print(x_test.shape)
print(y_test.shape)
