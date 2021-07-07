#Libraries
import pandas as pd
import numpy as np
from scipy.sparse import data

#Load data
datafile = pd.read_csv("categoricvalues.csv")
print(datafile)

#Categoric values
countrycodes = datafile.iloc[:0,1]
print(countrycodes)

#Encoding (change values as numeric)
from sklearn import preprocessing #Call Encoder

#Label Encoder
labelEnc = preprocessing.LabelEncoder()
countrycodes[:,0] = labelEnc.fit_transform(datafile.iloc[:,0])

# One Hot Encoding
oneHotEnc = preprocessing.OneHotEncoder()
countrycodes[:,0] = oneHotEnc.fit_transform(datafile.iloc[:,0])

