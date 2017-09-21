# Section 1: course introduction
## Application of ML
1. Face recognition in FB
2. `Kinect` use `Randon forest` to recognize people movement
3. Virtual Reality (VR) headset
4. Siri, Cortana...
5. Robot dog use `Rainforcement learning`
6. Facebook Ads
7. Amazon, Netflix use `Recomendation system`
8. Medicine
9. Use in spaces to recognize an area of the world to map
10. Use in robot on Mars :D

## ML is the future
Data that human created is really big. About 130 exabytes till 2005.

## Tools
1. R: `R studio`
2. Python: `Anaconda`

# Section 2: Data Preprocessing
## Get the dataset
Course data: https://www.superdatascience.com/machine-learning/

## Importing libraries
- Python libraries
    - numpy:
        - `import numpy as np`
        - Do the mathematic operation
    - matplotlib.pyplot:
        - `import matplotlib.pyplot as plt`
        - Drawing lib
    - pandas:
        - `import pandas as pd`
        - use to import dataset and manage dataset
    
- R libraries

## Importing the dataset
- In Python index start from 0, in R index start from 1
- Python code
```
import pandas as pd
dataset = pd.read_csv('Data.csv')
# Read all lines
# Read all column except the last column
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
```

- R code
```
dataset = read.csv('Data.csv')
```

## Missing data
- Python code: using `Imputer` of `sklearn.preprocessing` to preprocess the data
```
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3]) # upper bound is excluded -> get collumn 1 and 2
X[:,1:3] = imputer.transform(X[:,1:3])
```
- R code: the average function in R is `ave()`
```
# Takecaring of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
```

## Categorical data
- Encode the text value in dataset to number
- Python code
```
# Encode categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# First use LabelEncoder to encode label to number
encoder_X = LabelEncoder()
X[:,0] = encoder_X.fit_transform(X[:,0])

# Second use OneHotEncode to separate each category to one coloumn
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

# Last encode the dependance variable
encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)

```
- R code
```
# Transform categorical column
# Use factor function to transform categorical data
# ? how to one hot encode
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Germany', 'Spain'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes','No'),
                           labels = c(1,0))
```

## Split dataset to Training set and Test set
- Python code: Using sklearn.cross_validation sublib `train_test_split`
``` 
# Splitting the dataset into the Training set and Test set
# random_state => number of random samples
from sklearn.cross_validation import train_test_split
#X_train, X_test = train_test_split(X, test_size=0.2, random_state = 0)
#y_train, y_test = train_test_split(y, test_size=0.2, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
- R code: install `caTools` to split tranning set and test set
``` 
# Splitting the dataset into the Training set and Test set
# Install caTools if not exists
# install.packages('caTools')
# Load caTools if not loaded
library(caTools)
# Set seed. Choose any number we need
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```

## Feature scaling
- Two method for feature scaling: standardisation use deviation, normalisation use mean value
    - Mean: x_norm = (x - min_x) / (max_x - min_x)
    - Standard deviation: x_stand = (x - mean) / standard_deviation
- Depends on context we need scaling the dummy variables or not 
- Python: using StandardScaler to do feature scaling
``` 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Depend on context we do feature scaling for dummy variables or not
# Fit the traning set first and transform
X_train = sc_X.fit_transform(X_train)
# Use the traning set scaling and transform to the test set
X_test = sc_X.transform(X_test)
# Depend on context we do feature scaling for dependance variables or not
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
```
- R: using `scale` function
``` 
# Feature Scaling
# For colum that categorical trasnformed by factor function, these are not numeric so can't be scale
# example code: training_set = scale(training_set)
training_set[,2:3] = scale(training_set[,2:3])
# example code: test_set = scale(test_set)
test_set[,2:3] = scale(test_set[,2:3])
```

## Change working directory
- Python
``` 
import os
os.getcwd()
os.chdir('*****MachineLearning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression')
```
- R
```
getwd()
setwd('*****MachineLearning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression')
```

# Section 3: Regression test
Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

Some Machine Learning Regression models:
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Support Vector for Regression (SVR)
- Decision Tree Classification
- Random Forest Classification

# Section 4: Simple linear regression
## Intuition
- trực giác, sự hiểu biết qua trực giác; khả năng trực giác, điều (hiểu qua) trực giác
- `y = b0 + b1 * x1`
    - y: dependent variable
    - x: independent variable
## Python code
- In simple linear regression, python library take care the feature scaling job for us -> no need to implement feature scaling
- Use `LinearRegression` from `sklearn.linear_model` to do linear regression
```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
## R code
- Simple linear regression in R auto support feature scaling -> no need to implement feature scaling
- Use `lm` (linear modal) function in R to do the simple linear regression
- Use `summary(regressor)` function to see statistically info of the modal
- Use `predict` function to predict the test set
``` 
# Fitting the simple linear regression to Traning set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predict the Test set
y_pred = predict(regressor, newdata = test_set)
```
- Use `ggplot2` library to visualising the data set.
- When install `ggplot2` got error `tar: Failed to set default locale` . Here the workaround
``` 
 system('defaults write org.R-project.R force.LANG en_US.UTF-8')
```
- `ggplot` + `geom_point` -> draw point
- `ggplot` + `geom_line` -> draw line
- `ggplot` + `ggtitle` -> set title
- `ggplot` + `xlab` -> set X axis label
- `ggplot` + `ylab` -> set Y axis label