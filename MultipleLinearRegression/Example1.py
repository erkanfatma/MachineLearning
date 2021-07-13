# HUMIDITY PREDICTION

#Libraries 
import pandas as pd
import numpy as np


#Load data
df = pd.read_csv("tennis.csv")
#print(df)

#getting data info
#print(df.info()) #data info
#print("Length : ", len(df)) #number of rows

#Encoding
from sklearn.preprocessing import LabelEncoder
encoded_df = df.apply(LabelEncoder().fit_transform)
print(encoded_df)

#One Hot Encoding for outlook column
outlook = encoded_df.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print("Outlook")
print(outlook)

#Dataframe
outlook_result = pd.DataFrame(data= outlook, index=range(14), columns=['o','r','s'])
result = pd.concat([outlook_result, df.iloc[:,1:3]],axis = 1)
result = pd.concat([encoded_df.iloc[:,-2:], result], axis=1)
print("Result")
print(result)
 
# #Divide Data for Test and Train
from sklearn.model_selection import train_test_split
#result.iloc[:,:-1] = bağımsız değişken 
#result.iloc[:,-1:] = bağımlı değişken
x_train, x_test, y_train, y_test = train_test_split(result.iloc[:,:-1], result.iloc[:,-1:], test_size=0.33, random_state=0)
print("Y test")
print(y_test)

#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Y prediction")
print(y_pred)

#Backward Elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = result.iloc[:,:-1], axis = 1)

#calculating p-values
X_l = result.iloc[:,[0,1,2,3,4,5]].values #needed columns (except humidity)
X_l = np.array(X_l, dtype=float)
r_ols = sm.OLS(endog= result.iloc[:,-1:], exog= X_l)
r = r_ols.fit()
print(r.summary())

#removing x1 
result = result.iloc[:1:]
X_l = result.iloc[:,[0,1,2,3,4]].values #needed columns (except humidity)
X_l = np.array(X_l, dtype=float)
r_ols = sm.OLS(endog= result.iloc[:,-1:], exog= X_l)
r = r_ols.fit()
print(r.summary())

#Remove first column from x_train
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

#no need to re-regression but need to train
regressor.fit(x_train, y_train)

y_pred= regressor.predict(x_test)
print("New y prediction")
print(y_pred)