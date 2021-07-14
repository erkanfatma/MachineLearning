# Support Vector Regression
First show up for clustering.

Seperate Linearly seperable groups. A margin definition comes here. The aim is to find line that has max margin.  (SVM)

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svm_graph.png)

(SVR): to get maximum point margin value. possible to draw more than one line, select the line contains max margin (line that contains max data point inside of itself.)

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svr_graph.png)

In the example a margin defined as Ɛ. It is seen +Ɛ and -Ɛ values that margin place top and bottom. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svr_example.png)

First functions is linear regression. The formula shows 

* y = ax + b.

Second function is non linear regression. Non linear functions(K - kernel function). X value is not just a value, it can be a function in non linear SVR. 

* y = a.f(x) + b

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svr_definitions.png)

Aim is to find different models to draw a line close to data points. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svr_functions.png)

### Difference Between Models
Different models finds different lines. In the graph RBF is the best model. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/SupportVectorRegression/img/svr_models.png)

SVR defines margin values and select the margin which contains max data points. If there are more than one line, it selects the min margin line that contains max data points. Different functions can be used in this method such as linear, non linear, polynomial, gaussian radial basis function, exponential. 