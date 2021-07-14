# Multiple Linear Regression

Prediction  model when more then one feature exist in the dataset. First step is to find useful independent variables.

## Simple Linear Regression
General formula is: y = ax + b

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/simplelinearregression.png)

Simple linear regression contains an error rate inside of it. 

Example :  sales = α + ß(month) + ε

Example graph of simple linear regression:

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/slr_graph.png)

## Multiple Linear Regression 
There are different variables that is expressed with ß(i).  ß(0) is a constant and an error rate is shown.  

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/multiplelinearregression.png)

Example : length = ß(0) + ß(1)*weight + ß(2)*age +ß(3)* shoe_size + ε

Example graph of multiple linear regression:

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/mlr_graph.png)

Every point(P) in the graph depend on 3 different variables. formula of 'a' achieved with x,y,z values. The line of the model contains an error rate. 

P.S. : What about model with 4 independent variables? It can be calculated but not shown in the 3D graph.

### Dummy Data
A variable that states another  variable. 
In encoding a column values translated columns that have 0 and 1 values. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/dummyvariable.png)

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
    - with heuristic approach
    - include all variables if variable selection done and ensure of the selection. e.g. 10 variables exist in dataset and all of them make sense for the analysis.
    - if obligation exist. e.g. to evaluate success rate between variables and bank scores. 
    - for discovery, pre-understanding before usage of other selection methods. to estimate effects about variables to system.
2. **Backward Elimination** 
    - stepwise approach
    - At the beginning a dataset is huge and after eliminations some columns,variables, remove and the number of columns decrease. 
    
    ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/backwardelimination.png)

    - Significance Level (SL): success level like p value. 
    1. First define a significance level and then find variables (columns) that provide this significance level. 
    2. Then, build a model to understand significance level such as multiple linear regression. 
    3. Then you calculate p values in the regression for every variable. Looking at the variable that has maximum p value, If the p value is greater than SL (P > SL), then control every variables step by step (continue to step 4). If the p value is smaller than SL (P < SL), then finish up the machine learning.
    4. Remove the variable that has maximum p values in the selection.
    5. Update machine learning and continue to step 3.
    6. Finish up the machine learning.

        ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/backwardelimination_algorithm.png)
        
3. **Forward Selection**
    - stepwise approach
        
        ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/forwardselection.png)

    1. Define a significance level.
    2. Define a model using all variables.
    3. Next, calculate p values in the model for every variable. Look at the variable which has minimum value.
    4. Keep the variable that has minimum p-value. And select a new variable from the dataset and add the keeping value to system.
    5. Update machine learning and then select a new variable to continue. If the P < SL, then continue to select new variables. If P > SL then ends machine learning.

      ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/forwardselection_algorithm.png)
      
4. **Bidirectional Elimination**
    - stepwise approach
    - combination of forward selection and backward elimination

     ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/bidirectionalelimination.png)

     1. Define SL
     2. Define a model using all variables.
     3. Calculate p-value for all variables. Select the variable that has minimum p variable.
     4. Variable that selected in the step 3 which has the minimum p-value is holding and all other variables is added to system and the variable with minimum p-value is stored in system.
     5. Variables which P value smaller than SL is staying in the system and old variables not removed from the system.
     6. Finish up machine learning.

     ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/bidirectionalelimination_algorithm.png)

    - P.S. In some cases more than one significance levels can be define for one of them is used for forward step, another one is used for backward step. 
5. **Score Comparison**
    - with heuristic approach
    1. defining a success criterion.
    2. build all possible models. If there are 2 variables then, only 1 model exist. If there are 3 variables exist, then 3 models exist for machine learning. If n variables exist, model number is (2^n) -1 , +1.

    ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/numberofmodels.png)

    3. 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/allmethods.png)


### File Orders 
1. Data Preparation for Multiple Variables

data preprocessing. 

2. MultipleLinearRegression

Dividing data into parts as train and test. Implementing is same with linear regression but the difference is that x_train includes more than one variables.

3. Backward Elimination

to find relation between dependent and independent variables.
to find ß(0) value 
add a array as column with 22 rows, we add 22 times 1 because multiplier = 1 means 1 * ß(0)

```
X = np.append(arr = np.ones(22,1).astype(int), values = df, axis = 1)
```

The code meaning is np.ones creates array with 1's. and it contains 22 elements.

The type of the matrice is integer. (**astype(int)**)

**values = df** : append the array to df array.

**axis = 1** : add as a column. If axis = 0, then it adds a row to the array. 

**Calculating p-value**
To select variables:

```
X_l = conc_lr.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
```

Getting statistical models: to find model between length and variables, another saying calculating relation between variables and length one by one 

```
model = sm.OLS(length, X_l).fit() #getting statistical models
```

Print model: 

```
print(model.summary()) 
```

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/backwardelimination_result.png)

Analyzing P > |t|. where p value is smaller is better. for this data, x5 has a problem. Because in backward elimination we need smaller p values.
Solution is to eliminate 4th element in the array: 

```
X_l = conc_lr.iloc[:,[0,1,2,3,5]].values
```
Output: 

![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/PREDICTION/MultipleLinearRegression/img/backwardelimination_result2.png)


### Difference Between OneHotEncoding and LabelEncoding
Encoders using when transforming categoric variables to numberic variables. 

One hot encoding is useful when there are more than two different value exist in the column. For instance, outlook contains sunny, rainy and overcast. 3 different variables exist. Encoding them as 0-1 values not meaningful for these columns. We need to place it 3 different columns in the dataset.

Label encoding is useful when there are 2 different value exist in the column. For example, if the value is yes-no or true-false. It is sufficient to transform. 