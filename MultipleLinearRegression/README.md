# Multiple Linear Regression

Prediction  model when more then one feature exist in the dataset. First step is to find useful independent variables.

## Simple Linear Regression
General formula is: y = ax + b

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/MultipleLinearRegression/img/simplelinearregression.png)

Simple linear regression contains an error rate inside of it. 

Example :  sales = α + ß(month) + ε

Example graph of simple linear regression:

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/MultipleLinearRegression/img/slr_graph.png)

## Multiple Linear Regression 
There are different variables that is expressed with ß(i).  ß(0) is a constant and an error rate is shown.  

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/MultipleLinearRegression/img/multiplelinearregression.png)

Example : length = ß(0) + ß(1)*weight + ß(2)*age +ß(3)* shoe_size + ε

Example graph of multiple linear regression:

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/MultipleLinearRegression/img/mlr_graph.png)

Every point(P) in the graph depend on 3 different variables. formula of 'a' achieved with x,y,z values. The line of the model contains an error rate. 

P.S. : What about model with 4 independent variables? It can be calculated but not shown in the 3D graph.

### Dummy Data
A variable that states another  variable. 
In encoding a column values translated columns that have 0 and 1 values. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/MultipleLinearRegression/img/dummyvariable.png)

Data variable is an important feature when data preprocessing. For example if you want to change gender to numeric variable or country code to numeric variable. Think about plate number, then this data is dummy variable. 

For instance, change birth locations to city codes like İstanbul to 34 or change where they come from to city codes. Then this data comes to dummy variable. Both two variables located in the dataset at the same time is a risk for the model. Because every column effect machine learning algorithm, so selected algorithm is significant. 

If a column encoded, then both of the columns, original column and encoded column, can not be used at the same time. In the above example, just female column can be selected to use in the model, then male and gender columns must be removed. In another case, if we select to use male and female columns in the algorithm, we have 2 columns that refers the same thing. In this kind of usage, the model can also be changed. The best is to choose one of them. Gender is binomial.

If a direct connection exist in encoded columns, it is significant to select dummy data. If there are more than one dummy data, then selection of dummy data is essential. For example if a birth place is encoded like gender column, then we select all of the encoded columns to use because there is no information about where the place is. If Izmir is 0 then this means not Ankara is 1. There is no clear information there. Birth place is polynomial.

### p-value
Probability value.
Usefull in null hypothesis (H0).

#### H0 (null hypothesis)
 Example in a cookie factory. A cookie is generally 30 grams, think that if more than 30 grams cookies, less than 30 grams cookies exist, if exist how many of them and so on.

#### H1
Reverse of H1. Example if time span of the lecture increase, the success rate is not increases always. to fail hypothesis. 

p-value: With how many examples to fail hypothesis. Generally selected as 0.05.

If p decreases, failure of H0 will increases. 
If p increases, failure of H1 will increases. 

### Variable Selection
While selecting variables, think that if every variable affects the dependent value(y) at the same rate? Or some variables do not effect the value. Is it necessary to select all the variables? Different approaches exist for the question

1. **Include All Variables**
    
2. **Backward Elimination** 

3. **Forward Selection**

4. **Bidirectional Elimination**

5. **Score Comparison**