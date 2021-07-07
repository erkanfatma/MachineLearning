#Libraries
import pandas as pd
import numpy as np
from scipy.sparse import data

#Load data
datafile = pd.read_csv("simpledata.csv")

#Numeric values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')

#find missing values in age column
ages = datafile.iloc[:,1:4].values  # iloc = integer location

imputer = imputer.fit(ages[:,1:4]) # learn mean values of columns

ages[:,1:4] = imputer.transform(ages[:,1:4]) # change nan values


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

#DATAFRAME
result = pd.DataFrame(data = countrycodes, index= range(22), columns= ['fr', 'tr', 'us'])

result2 = pd.DataFrame(data = ages, index = range(22), columns=['length', 'weight', 'ages'])

gender = datafile.iloc[:,-1].values
print(gender)
result3 = pd.DataFrame(data = gender, index = range(22), columns=['cinsiyet'])

# Combine dataframes 
res = pd.concat([result, result2], axis = 1) # axis = 1 , in every index columns are added 
print(res)

res2 = pd.concat([res, result3], axis=1)
print(res2)



