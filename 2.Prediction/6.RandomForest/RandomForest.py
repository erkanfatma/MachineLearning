#Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Load file
df = pd.read_csv("PolynomialRegression\\salaries.csv")

#data frame (slice)
x = df.iloc[:,1:2]
y = df.iloc[:,2:]

#numpy array transformation
X = x.values 
Y = y.values 
 
# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0) # number of estimators = how many decision tree will be drawn

rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

Z= X + 0.5
K = X - 0.4
plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X), color = 'blue')
plt.plot(X, rf_reg.predict(Z), color = 'green')
plt.plot(X, rf_reg.predict(K), color = 'yellow')

plt.show()
