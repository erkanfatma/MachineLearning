from numpy.matrixlib.defmatrix import asmatrix
from pandas.core.accessor import register_series_accessor
from DataPreparation import *

#Multiple Linear Regression -- Gender prediction
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train) # learn y_train values from x_train and generate model between them (%67 for training)

y_pred = regressor.predict(x_test)
# print('Y Test')
# print(y_test)
# print('Y Prediction')
# print(y_pred)


#Multiple Linear Regression -- Length prediction
length = res2.iloc[:,3:4].values
#print(length)
left = res2.iloc[:,:3]
right = res2.iloc[:,4:]

#Removing length column and combining others
conc_lr = pd.concat([left, right], axis=1)

x_train, x_test, y_train, y_test = train_test_split(conc_lr, length, test_size=0.33, random_state= 0)
# print(x_train)
# print(y_train)

r2 = LinearRegression()
r2.fit(x_train, y_train)
y_pred = r2.predict(x_test)
#Comparing real vals and predictions
# print(y_test)
# print(y_pred)