#Libraries
import pandas as pd 
import numpy as np

#Load file
df = pd.read_csv("salaries.csv")

#data frame (slice)
x = df.iloc[:,1:2]
y = df.iloc[:,2:]

#numpy array transformation
X = x.values
Y = y.values

#With Linear Regression 
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(x,y)
#visualization
import matplotlib.pyplot as plt
plt.scatter(X,Y, color='orange')
plt.plot(x,le.predict(X), color='green')
plt.show()

#Polynomial Regression - non linear model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #define function degree #polynomial regression definition , with degree 4
x_pol = poly_reg.fit_transform(X) #transform
print(x_pol) 
# -- Polynomial with linear regression - before applying linear regression transform data to polynomial 
lin_reg = LinearRegression()
lin_reg.fit(x_pol,y)
#visualization
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()
# --

#Predictions
print(le.predict([[11]]))
print(le.predict([[6.6]]))
print(lin_reg.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg.predict(poly_reg.fit_transform([[11]])))