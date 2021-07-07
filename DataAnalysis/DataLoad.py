# Libraries
import pandas as pd #data operations
import numpy as np # comprehensive mathematical functions
import matplotlib.pyplot as plt #visualization

# Load data
datafile = pd.read_csv('simpledata.csv')
print(datafile)

# Data Preprocessing
lenghts = datafile[['length']]
print(lenghts)

weight_and_gender = datafile[['weight', 'gender']]
print(weight_and_gender)

