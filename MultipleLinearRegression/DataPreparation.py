#Libraries
import pandas as pd
import numpy as np

#Load data
df = pd.read_csv("\\DataAnalysis\\simpledata.csv")
#print(df)

#Encoding of country
country = df.iloc[:,0:1].values
#print(country)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(df.iloc[:,0])
#print(country)
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
#print(country)

#Encoding of gender
gender = df.iloc[:,-1:].values
#print(gender)
from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
gender[:,-1] = le2.fit_transform(df.iloc[:,-1])
ohe2 = preprocessing.OneHotEncoder()
gender = ohe2.fit_transform(gender).toarray()
#print(gender)

age = df.iloc[:,1:4].values

result = pd.DataFrame(data=country, index=range(22), columns=['fr','tr','us'])
result2 = pd.DataFrame(data=age, index = range(22), columns = ['length', 'weight', 'age'])
result3 = pd.DataFrame(data=gender[:,:1],index=range(22), columns = ['gender'])

res = pd.concat([result, result2], axis = 1)
#print(res)

res2 = pd.concat([res, result3], axis= 1)
#print(res2)


#Divide Data for Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(res, result3, test_size=0.33, random_state=0)

#Data Scaling -- standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# print("------- x train")
# print(x_train)
# print("----------- x test")
# print(x_test)

# print("----------- y train")
# print(y_train)
# print("----------- y test")
# print(y_test)