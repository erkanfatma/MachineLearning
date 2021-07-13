from DataPreparation import *
from MultipleLinearRegressionF import *
import numpy as np
import pandas as pd
# measuring success of the model
import statsmodels.api as sm

#to find ß(0) value -- add a array as column with 22 rows, we add 22 times 1 because multiplier = 1 means 1 * ß(0)
X = np.append(arr = np.ones((22,1)).astype(int), values = conc_lr, axis = 1)
#print(X)

#calculating p-values
X_l = conc_lr.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(length, X_l).fit() #getting statistical models - find model between length and variables
print(model.summary()) # Analyzing P > |t|. 

#Remove x5
X_l = conc_lr.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(length, X_l).fit()
print(model.summary())

#To get more significance model 
X_l = conc_lr.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(length, X_l).fit()
print(model.summary())
