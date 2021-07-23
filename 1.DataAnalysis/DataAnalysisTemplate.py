#LIBRARIES
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data

#DATA PREPROCESSING

#Load Data
df = pd.read_csv('simpledata.csv')
print(df)

#Missing values - to impute
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy='mean')

ages = df.iloc[:,1:4].values
print(ages)

imputer = imputer.fit(ages[:,1:4])
ages[:,1:4] = imputer.transform(ages[:,1:4])
print(ages)

#Encoder -- transform categoric values to numberic values
countries = df.iloc[:,0:1].values
print(countries)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(df.iloc[:,0])
print(countries)

ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)

#Transform Numpy arrays to dataframe
result = pd.DataFrame(data= countries, index = range(22), columns=['fr','tr','us'])
result2 = pd.DataFrame(data = ages, index= range(22), columns=['length', 'weight', 'age'])
gender = df.iloc[:,-1].values
result3 = pd.DataFrame(data = gender, index= range(22), columns=['gender'])

#Concatenate dataframes
res = pd.concat([result, result2], axis= 1)
res2 = pd.concat([res, result3], axis= 1)

#Divide Data for Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(res, result3, test_size=0.33, random_state=0)

#Data Scaling -- standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)