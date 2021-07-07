## Data Analysis

### Data Load
Data load operation done with pandas.

```
import pandas as pd

datafile = pd.read_csv('simpledata.csv')
```

### Data Types
![alt text](https://github.com/erkanfatma/MachineLearning/blob/main/DataAnalysis/img/datatypes.png)

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


2. Categoric values