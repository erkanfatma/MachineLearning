#Libraries
import pandas as pd 
import numpy as np

#Load data
datafile = pd.read_csv('missingvalues.csv')
print(datafile)

#Missing values

#Numeric values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')

#find missing values in age column
ages = datafile.iloc[:,1:4].values  # iloc = integer location
print(ages)

imputer = imputer.fit(ages[:,1:4]) # learn mean values of columns

ages[:,1:4] = imputer.transform(ages[:,1:4]) # change nan values
print(ages)


#Non numeric values