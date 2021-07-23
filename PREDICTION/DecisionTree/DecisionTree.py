#Libraries
import pandas as pd 
import numpy as np

#Load file
df = pd.read_csv("PolynomialRegression\\salaries.csv")

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

#Data Scaling - important for SVR
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scale = sc1.fit_transform(X)

sc2 = StandardScaler()
y_scale = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

#SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf') #rbf = kernel function. for more kernel functions = https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
svr_reg.fit(x_scale, y_scale)

plt.scatter(x_scale,y_scale, color='pink')
plt.plot(x_scale,  svr_reg.predict(x_scale), color='blue')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)

dt_reg.fit(X, Y) #learn Y from X 

plt.scatter(X,Y, color = "black")
plt.title("Decision Tree")
plt.plot(x, dt_reg.predict(X), color = "red") #dt_reg.predixt(X) -> prediction for each X val 

#for each values the same point is shown in the three because (billing) each values located in the same group. Decision tree goes to the same leaf for these values.
Z = X + 0.5
K = X - 0.4
plt.plot(x, dt_reg.predict(Z), color = "green")
plt.plot(x, dt_reg.predict(K), color = "yellow")

plt.show()

print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))