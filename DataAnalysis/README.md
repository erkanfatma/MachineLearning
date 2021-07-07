## Data Analysis

### Data Load
Data load operation done with pandas.

```
import pandas as pd

datafile = pd.read_csv('simpledata.csv')
```

### Data Types
![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/DataAnalysis/img/datatypes.png)

Data types contains 2 different categories: categoric and numeric.
1. Categoric
    e.g. gender, educational level, licence plate code.
    no relation like greater or smaller 

    1. Nominal
        no order.
        e.g. model of cars, brands
        1. Binomial
        2. Polynomial
    2. Ordinal
        ordinal comes from order.
        e.g. licence plate code, door number

2. Numeric
    e.g. age, weight

    1. Ratio
        can be proportion
        e.g. age
    2. Interval
        within a range
        e.g. temperature 

### Missing Values

1. Numeric values
To fill the missing values: 
    1. assign a constant numeric value
    2. generate mean value and assign to missing values
    
        ```
        imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean')
        ```
        Missing values find with the numpy.nan that imputes nan values
        Strategy that to replace nan with mean 

        
        ```
        ages = datafile.iloc[:,1:4].values  # iloc = integer location
        ```
        Find missing values in age column


        ```
        imputer = imputer.fit(ages[:,1:4])
        ```
        Get mean value(Fit function used to train generally)


        ```
        ages[:,1:4] = imputer.transform(ages[:,1:4]) 
        ```
        Change nan values

P.S. Fit that machine learning learns. Transform that machine learning perform. 

2. Categoric values
Some of the machine learning algorithms need numeric values to work with. In this case, the categoric values transform to the numberic values.

    - Nominal
        - Binomial
            e.g. gender
            2 different values (0,1)
        -  Polynomial
            e.g. country codes
            every label transform as a column and assigned as 0,1
            ![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/DataAnalysis/img/polynomialvalues.png)

    - Ordinal 
        Changing it to numeric values



    