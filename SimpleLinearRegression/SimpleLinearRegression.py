#Libraries
import pandas as pd
import numpy as np

#Load data
df = pd.read_csv("sales.csv")
#print(df)

#divide data into two parts -> train & test
months = df[['months']] #independent values
#print(months)
sales = df[['sales']] #dependent values
#print(sales)
sales2 = df.iloc[:,:1].values
#print(sales2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=0.33, random_state=0)

#Data scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


#Modelling with Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train) # generates a model (relationship) between X data and Y data using train

predictionResult = lr.predict(X_test) # predict Y using X_test data
#print("Y test")
#print(Y_test)
print("Prediction")
print(predictionResult)

#Without scaling
lr2 = LinearRegression()
lr2.fit(x_train, y_train)
predictionResult2 = lr2.predict(x_test)
#print("y test")
#print(y_test)
print("Prediction without scaling")
print(predictionResult2)

#Visualization
import matplotlib.pyplot as plt
x_train = x_train.sort_index() #order x data
y_train = y_train.sort_index() #order y data

plt.plot(x_train, y_train)
plt.plot(x_test, lr2.predict(x_test))
plt.title("Sales By Months")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()
