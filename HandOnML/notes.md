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
- Python draw functions
    - `plt.scatter` to print points
    - `plt.plot` to draw lines
    - `plt.title` to set the tittle
    - `plt.xlabel` to set x label
    - `plt.ylabel` to set y label
    - `plt.show` to show the figure

## R code
- Simple linear regression in R auto support feature scaling -> no need to implement feature scaling
- Use `lm` (linear model) function in R to do the simple linear regression
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

# Section 5: Multiple linear regression
## Multiple linear regression intuition
- Multiple linear regression hypothesis function
```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
b0: constant
b1, b2, ..., bn: coeficients
```

- Linear regression
    - Linearity
    - Multivariate normality
    - Independence of errors
- Dummy variable:
    - the variable created by using OneHotEncoder on categorical variables
    - We should not implement all of dummy variable at once. It will cause bad result (Dummy variable trap)
    - If we have `n` dummy variables -> use `n-1` variable in Hypothesis function

## Steps to build a modal
- Feature selection
    - Remove some meaningless variables (more light-weight modal)
    _ Some day we have to explain the meaning of all variable. We need to find only necessary features.
- 5 method to building a modal
    1. All in: use all features if:
        - we have prior knowledge of all variables, OR
        - we have to build the modal (on behave of boss, leader...), OR
        - prepare for `Backward elimination`
    2. Backward elimination (Can be specified as `stepwise regression`)
        - Step 1: select a significance level to stay in the model (Ex: SL = 0.05)
        - Step 2: fit the full modal with all possible predictors
        - Step 3: consider the predictor with the highest P-value. If P>SL, go to Step 4, otherwise go to FINISH
        - Step 4: remove the predictor
        - Step 5: fit the model without this variable *
        - __-> It's the fastest method at all__
    3. Forward selection (Can be specified as `stepwise regression`)
        - Step 1: select a significance level to stay in the model (Ex: SL = 0.05)
        - Step 2: Fit all simple regression model `y ~ xn`. Select the one with the lowest P-value
        - Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) we already have
        - Step 4: Consider the predictor with the lowest P-value. If P < SL, go to Step3, otherwise go to FINISH
        - FINISH: keep the previous modal
    4. Bidirectional elimination (Can be specified as `stepwise regression`)
        - Step 1: Select a significance level to enter and to stay in the modal. Eg: SLENTER = 0.05, SLSTAY = 0.05
        - Step 2: Perform the nextstep of `Forward Selection` (new variable must have P < SLENTER to enter)
        - Step 3: Perform ALL steps of `Backward elimination` (old variables must have P < SLSTAY to stay)
        - Step 4: No new variables can enter or exit
        - FINISH: Our modal is ready
    5. Score comparison (All possible modal)
        - Step 1: Select a criterion of goodness of fit
        - Step 2: Construct all possible regression modals: 2^n -1 total combination
        - Step 3: Select the one with best criterion
        - FINISH: we have the modal
        - __-> Not good approach because if we have 10 variables -> we need to try 1023 possible modal__

## Multiple linear regression in Python code
- Remember to encode the independence variable and avoid Dummy variable trap
- Use `sklearn.linear_model` and sublibrary `LinearRegression`

## Multiple linear regression in R code
- Similar with simple linear regression. We use `lm` model to do the regression

## Backward elimination in Python
- We can use `statsmodels.formula.api` to implement backward elimination
- The model we use in OLS (Ordinary least square)
```
# Building the optimal modal using Backward elimination
import statsmodels.formula.api as sm
# Add 1 column for constant b0 in equation: y = b0 + b1*x1 + ... + bn*xn
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# Ordinar least square modal using in backward elimination
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove third variable. P = 0.990 > 0.05
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove second variable. P = 0.940 > 0.05
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove third variable. P = 0.604 > 0.05
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove third variable. P = 0.060 > 0.05
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# The model is ok now => Profit is highly depend on R&D Spend
```
## Backward elimination in R
- We still use `lm` function to do regression in R. But the `summary` function can show the statistically significant result
```
# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# Remove State (The lm lib auto do the encode categorical data from State to State1, State2, State3 and auto ommit State 1)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
# Remove Administration P = 0.602 > 0.05 
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)
# Remove Marketing.Spend P = 0.06 > 0.05
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
```

# Section 6: Polynomial linear regression
## Polynomial linear regression intuition
- Equation:
```
y = b0 + b1*x1 + b2*x1^2 + ... + bn*b1^n
```
- We still call the equation is a linear because the coefficient is `Linear`
- The model is `linear` when the equation has `linear` coefficients
## Polynomial linear regression in Python
- User `PolynomialFeatures` to transform dataset to polynomial dataset
- After transformation we can use linear regression to the transformed dataset
``` 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```
## Polynomial linear regression in R
- Add extra polynomial variable and do the linear regression for new dataset
``` 
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
```

# Section 7: Support Vector Machine Regression (SVR)
## SVR in python
- Use `SVR` sub library of `sklearn.svm`
- Make sure we do the feature scaling when using SVR
``` 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)
```
- Do the inverse transform when get the prediction result
```
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
```
## SVR in R
- Make sure we install `e1071` to import `SVR` function
```
install.packages('e1071')
```
- Use `svm` function with `type='eps-regression'` to make the regressor
``` 
library('e1071')
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')
```
- In R we dont need to do the feature scaling with `svm` lib

# Section 8: Decision Tree Regression
## Decision Tree Regression intuition
- CART
    - CA: Classification Trees
    - RT: Regression Trees
- Decision Tree: split ranges of dependence variables to parts
- The predict value of independence variable will be the average values of the part of dependence variables that associated with its
## Decision Tree Regression in Python
- No need do feature scaling
- Use `DecisionTreeRegressor` in `sklearn.tree`
- `DecisionTree Model` isn't continue. Its prediction based on the average number of each variable ranges. Other models is continue.
- In `1-d` this model is not interesting. But with more dimension dataset it's very powerful.
```
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
```
## Decision Tree Regression in R
- No need do feature scaling
- Install `rpart` package to use decision tree in R `install.packages('rpart')`
```
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))
```

# Section 9: Random Forest Regression
## Random forest intuition
- `Random forest model` is a team of `decision tree models`
- Ensemble learning -> take multiple times of a learning and get the accumulation result
- Steps:
    - Step1: Pick random K data points from the Training set
    - Step2: Build decision tree on these K data points
    - Step3: Choose the number Ntree of trees you want to build in Step1&2
    - Step4: For a new data point, make each one of our new Ntree trees predict the new value of Y to for the data point in question, and assign the new data point the average across all of the predicted Y values.

## Random Forest Regression in Python
- Using `RandomForestRegressor` library from `sklearn.ensemble` to do random forest regression
- Remember to choose the `n_estimators`. It's the number of time we do the decision tree models.
```
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0);
regressor.fit(X,y)
```