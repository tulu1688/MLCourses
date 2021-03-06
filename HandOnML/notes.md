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

## Some references
- [Machine Learning Algorithms: Which One to Choose for Your Problem](https://dzone.com/articles/machine-learning-algorithms-which-one-to-choose-fo)

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
- Python code: Using sklearn.model_selection sublib `train_test_split`
```
# Splitting the dataset into the Training set and Test set
# random_state => number of random samples
from sklearn.model_selection import train_test_split
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
        - Step 2: Construct all possible regression modals: 2<sup>n</sup> -1 total combination
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
- Equation: __y = b<sub>0</sub> + b<sub>1</sub>*x<sub>1</sub> + b<sub>2</sub>*x<sub>1</sub><sup>2</sup> + ... + b<sub>n</sub>*x<sub>1</sub><sup>n</sup>__
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

## Random Forest Regression in R
- Install `randomForest` library to do `Random Forest Regression`
- Use `randomForest` function after install library. The function's params
    - x: dataframe of dependence variables
    - y: vector of independence variable
    - ntree: number of tree to pick
```
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], # dataframe
                         y = dataset$Salary, # Vector
                         ntree = 10) # Number of trees to grow
```

# Section 10: Evaluating regression models performance
## R-squared intuition
```
SSres = (yi - yi^)2 -> min (SSres: sum of squared residual)
SStotal = (yi - yavg)2 -> (SStotal: total sum of squared)
R<sup>2</sup> = 1 - SSres/SStotal
```

- R<sup>2</sup> closer to 1, better our model. `Greater is better`
- Khi càng thêm predictor vào biểu thức của model -> R^2 sẽ càng giảm.
    - Ví dụ dataset có 5 independence variable. Dùng 4 independence variable có Rsquare lớn hơn dùng 5.

## Ajusted R-Squared intuition
- Relearn
```
Adj R^2 = 1 - (1-R^2)*(n-1)/(n-p-1)
p: Number of regressor
n: sample size
```

## Evaluating Regression Models Performance

__Example__ givens the model equations and the Adjusted R squared

1. Profit ~ R.D.Spend + Administration + Marketing.Spend + State => Adjusted R squared: 0.9452
2. Profit ~ R.D.Spend + Administration + Marketing.Spend => Adjusted R squared: 0.9475
3. Profit ~ R.D.Spend + Marketing.Spend => Adjusted R squared: 0.9483
4. Profit ~ R.D.Spend => Adjusted R squared: 0.9454

__Model 3__ là tốt nhất vì có Adjusted R squared lớn nhất.

# Section 11: Classification
- Classification models
    - Logistic Regression
    - K-Nearest Neighbors (K-NN)
    - Support Vector Machine (SVM)
    - Kernel SVM
    - Naive Bayes
    - Decision Tree Classification
    - Random Forest Classification

# Section 12: Logistic regression
## Logistic regression intuition
- sigmoid function: ___p = 1 / (1 + e<sup>-y</sup>)___
- We use sigmoid function because ___ln(p/(1-p)) = b0 + b1*x___
- Use logistic regression to predict probability
## Logistic regression in python
- Remember to do the feature scaling
- Use `LogisticRegression` library from `sklearn.linear_model` to do the Logistic Regression
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```
- Use `confusion_matrix` library from `sklearn.metrics` to compute confusion metrics
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

## Logistic regression in R
- Remember to do the feature scaling
- Use `glm` function with parameters:
    - family: binomial
    - data: training_set
```
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)
```

- Predicting with logistic regression
```
# Predicting the Test set results
#prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
prob_pred = predict(classifier, type = 'response', newdata = test_set[1:2])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the confusion matrix
cm = table(test_set[,3], y_pred)
```

- Install package `Elemstatlearn` to visualing the data
```
install.packages('ElemStatLearn')
```

# Section 13: K-Nearest neighbors (K-NN)
## K-NN intuition
- K-NN is non-linear classification
- Step 1: Choose the number K of neighbors
- Step 2: Take the K nearest neighbors of the new data point, according to the Euclidean distance
- Step 3: Among these K neighbors, count the number of data points in each category
- Step 4: Assign the new data point to the category where we counted the most neighbors
- We got the model

## K-NN classification in Python
- Use `KNeighborsClassifier` from `sklearn.neighbors` to implement K-NN in python

```
# Fitting the K-NN classifier to Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = 'minkowski', # metric: minkowski or euclidean
                                  p = 2)
classifier.fit(X_train, y_train)
```

## K-NN classification in R
- Use `class` library in R to implement K-NN

```
library(class)
y_pred = knn(train = training_set[,1:2],
             test = test_set[,1:2],
             cl = training_set[,3],
             k = 5) # Number of neighbors
```

# Support Vector Machine (SVM)
## SVM intuition
- SVM ~ maximum margin classifier

## SVM classification in Python
- Using `SVC` sub library in `sklearn.svm` to implement SVM classification

```
from sklearn.svm import SVC
# We can test some type of kernel to get the best kernel here
# classifier = SVC(kernel = 'rbf',  random_state = 0)
# classifier = SVC(kernel = 'poly', degree = 3, coef0 = 0.001, random_state = 0)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
```

## SVM classification in R
- Install `e1071` library to implement SVM classification
- Fit the classifier to the Training set

```
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')
```

- Predicting the Test set results

```
y_pred = predict(classifier, newdata = test_set[,1:2])
```

# Section 15: Kernel SVM
- Kernel function help SVM find the decision boundary
- The Gaussian RBF (Radial based function) Kernel: ___K(x,li) = e<sup>(-abs(x-li)^2/(2σ^2))</sup>___
    - li: landmark point
    - x: training set point
- Theses types are mostly used of kernel functions:
    - Gaussian RBF kernel
    - Sigmoid kernel
    - Polynomial kernel

# Section 16: Naive Bayes
## Bayes theorem and Naive Bayes intuition
```
P(A|B) = P(B|A)*P(A)/P(B)
```

## Naive Bayes classification in Python
- Using `GaussianNB` in `sklearn.naive_bayes` library to implement NaiveBayes classification

```
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## Naive Bayes classification in R
- Install `e1071` library to use `Naive Bayes` classification
- Function `naiveBayes`
- In NaiveBayes in R we need to encoding the target feature as factor (smth like categorical variables)

# Section 17: Decision Tree Classification

## Decision Tree Classification in Python
- use `DecisionTreeClassifier` in `sklearn.tree` to implement decision tree classification

```
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

## Decision Tree Classification in R
- In `Decision Tree Classification` in R we need to encoding the target feature as factor (smth like categorical variables)

```
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))
```

- Use `rpart` function to implement decision tree clasification

```
# install.packages('rpart')
library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[,1:2], type = 'class')
```

- Remember to use `type = 'class'`, rpart will know that we need to use it's classification feature


# Section 18: Random Forest Classification

## Random forest classification intuition
- Random forest is an ensemble learning method by containing numbers of decision tree classification
- Random forest usually is used in detect people movement

## Random forest classification in Python
- Use `RandomForestClassification` in `sklearn.ensemble`

```
# Fitting the classifier to Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, # set too much n_estimators -> overfiting
                                    criterion = 'entropy',
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test set result
y_pred = classifier.predict(X_test)
```

## Random forest classification in R
- Intall `randomForest` package to run classification

```
install.packages('randomForest')
```

- Remember to encode the target feature as factor (smth like categorical variables)

```
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))
```

- Remember to select the ntree = number of tree of decision tree

```
library(randomForest)
classifier = randomForest(x = training_set[,1:2],
                          y = training_set$Purchased,
                          ntree = 100)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[1:2])

# Making the confusion matrix
cm = table(test_set[,3], y_pred)
```

# Section 19: Evaluating classification models performance
## Confusion matrix
- Accuracy rate: Correct / Total
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Error rate: Wrong / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score = 2 * Precision * Recall / (Precision + Recall)
## Accuracy paradox
1. First scenario

```
    Predicted
     0     1
  -------------
0 | 9700   150
1 |  50    100
```

Accuracy rate: AR = (9700 + 100) / 10000 = 98%
2. Second scenario

```
    Predicted
     0     1
  -------------
0 | 9850   0
1 |  150   0
```

Accuracy rate: AR = 9850 / 10000 = 98.5%
3. Conclusion
- The accuracy rate in second scenario is better than in first scenario. But the model in second scenario is not better. It's worst, because no True value predicted.
- The accuracy rate is not really good to determine a model is good or not

## CAP curve
- CAP: Cumulative Accuracy Profile
- ROC: Receiver Operating Characteristic
- Actually dont really know CAP well

## ROC curve
- [Mục 4.2 - Receiver Operating Characteristic curve](https://machinelearningcoban.com/2017/08/31/evaluation/#-truefalse-positivenegative)
- Trong một số bài toán, việc tăng hay giảm FNR, FPR có thể được thực hiện bằng việc thay đổi một ngưỡng (threshold) nào đó. Lấy ví dụ khi ta sử dụng thuật toán Logistic Regression, đầu ra của mô hình có thể là các lớp cứng 0 hay 1, hoặc cũng có thể là các giá trị thể hiện xác suất để dữ liệu đầu vào thuộc vào lớp 1. Khi sử dụng thư viện sklearn Logistic Regression, ta có thể lấy được các giá trị xác xuất này bằng phương thức predict_proba(). Mặc định, ngưỡng được sử dụng là 0.5, tức là một điểm dữ liệu x sẽ được dự đoán rơi vào lớp 1 nếu giá trị predict_proba(x) lớn hơn 0.5 và ngược lại.
- Nếu bây giờ ta coi lớp 1 là lớp Positive, lớp 0 là lớp Negative, câu hỏi đặt ra là làm thế nào để tăng mức độ báo nhầm (FPR) để giảm mức độ bỏ sót (FNR)? Chú ý rằng tăng FNR đồng nghĩa với việc giảm TPR vì tổng của chúng luôn bằng 1.
- Một kỹ thuật đơn giản là ta thay giá trị threshold từ 0.5 xuống một số nhỏ hơn. Chẳng hạn nếu chọn threshold = 0.3, thì mọi điểm được dự đoán có xác suất đầu ra lớn hơn 0.3 sẽ được dự đoán là thuộc lớp Positive. Nói cách khác, tỉ lệ các điểm được phân loại là Positive sẽ tăng lên, kéo theo cả False Positive Rate và True Positive Rate cùng tăng lên (cột thứ nhất trong ma trận tăng lên). Từ đây suy ra cả FNR và TNR đều giảm.
- Ngược lại, nếu ta muốn bỏ sót còn hơn báo nhầm, tất nhiên là ở mức độ nào đó, như bài toán xác định email rác chẳng hạn, ta cần tăng threshold lên một số lớn hơn 0.5. Khi đó, hầu hết các điểm dữ liệu sẽ được dự đoán thuộc lớp 0, tức Negative, và cả TNF và FNR đều tăng lên, tức TPR và FPR giảm xuống.
- Như vậy, ứng với mỗi giá trị của threshold, ta sẽ thu được một cặp (FPR, TPR). Biểu diễn các điểm (FPR, TPR) trên đồ thị khi thay đổi threshold từ 0 tới 1 ta sẽ thu được một đường được gọi là Receiver Operating Characteristic curve hay ROC curve. (Chú ý rằng khoảng giá trị của threshold không nhất thiết từ 0 tới 1 trong các bài toán tổng quát. Khoảng giá trị này cần được đảm bảo có trường hợp TPR/FPR nhận giá trị lớn nhất hay nhỏ nhất mà nó có thể đạt được)

## More docs
- [Evaluating classification models](https://machinelearningcoban.com/2017/08/31/evaluation/#-truefalse-positivenegative)
- [TiepVuSu github](https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/33_evaluation/python/Evaluation%20methods.ipynb)

## Conclusion of Part 3 - Classification
- Choose model
    - If your problem is linear, you should go for Logistic Regression or SVM.
    - If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.
- Then from a business point of view, we would rather use:
    - Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.
    - SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.
    - Decision Tree when you want to have clear interpretation of your model results
    - Random Forest when you are just looking for high performance with less need for interpretation

# Section 20: Clustering
- Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.
- Some clustering models:
    - K-Means Clustering
    - Hierarchical Clustering

# Section 21: K-Mean clustering
## K-mean clustering intuition
- Step 1: choose the number K of clusters
- Step 2: Select random K points as centroid (Not necessarily from our dataset)
- Step 3: Assign each data point to the closest centroid -> that forms K clusters
- Step 4: Reassign each data point to the new closest centroid. If any assignment took place -> go to Step 4 or Finish
## K-mean random initialization trap
- K-mean++ can help us avoid the trap of randomly select initial centroids
## K-mean choosing the right number of clusters
- Compute WCSS and draw charts of WCSS and number of cluster
- Choose the value at corner of this L (elbow) shape -> optimal number of clusters

## K-means in python
- First we need to compute WCSS for each number of clusters and draw the "Elbow model" chart. Select the number of cluster in corner of our elbow
- Using `KMeans` from `sklearn.cluster` to cluster

```
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

- Parameters of KMeans
    - n_clusters: number of clusters
    - init: 'k-means++' help us not falling into random initialization trap
- After select the optimal number of cluster, we can re-cluster with k-means

## K-means in R
- First we need to compute WCSS for each number of clusters and draw the "Elbow model" chart. Select the number of cluster in corner of our elbow

```
# Using the elbow method to find the optimal number of clusters
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withins)
plot(1:10, wcss, type = 'b', main = paste('Clusters of clients'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')
```

- Use `kmeans` function to clustering
- Use clusplot to draw kmeans clusters

```
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualising the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Cluster of the clients"),
         xlab = "Annual income",
         ylab = "Spending score")
```

# Section 22: Hierarchical clustering
## Hierarchical clustering intuition
- Two types of hierarchical clustering approach: agglomerative & divisive
- Agglomerative Hierarchical Clustering
    - __Step 1__: Make each data point a single-point cluster -> that forms N clusters
    - __Step 2__: Take the two closest data points and make them on cluster -> that forms N-1 clusters
    - __Step 3__: Take the two closest clusters and make them on cluster -> that forms N-2 clusters
    - __Step 4__: Repeat __Step 3__ until there's only one cluster -> FINISH
- 4 options of distance between clusters
    - Option 1: Distance of closest points
    - Option 2: Distance of furthest points
    - Option 3: Average distance
    - Option 4: Distance between centroids

## Hierarchical dendograms
- Dendogram works like a memory of a HC algorithm
- [Refrence link](https://en.wikipedia.org/wiki/Dendrogram)
- Optimal number of clusters is number of intersections between the line pass the largest distance

## Hierarchical clustering in Python
- Using `scipy.cluster.hierarchy` to create dendogram

```
# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()
```

- User `AgglomerativeClustering` from `sklearn.cluster` to clustering
```
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
```

## Hierarchical clustering in R
- Using dendogram to find the optimal number of clusters

```
dendogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendogram,
    main = paste('Dendogram'),
    xlab = 'Customers',
    ylab = 'Euclidean distances')
```

- Using `cutree` function for `hierarchical model` to clustering
```
# Fitting the hierarchical clustering to mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)
```

# Section 22_b: DBScan
## DBScan in python
- Using `sklearn.cluster.DBSCAN` to cluster

```
# Using dbscan to find frequent data zones
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.2, # Maximum distance between two samples
                min_samples = 5,
                metric = 'euclidean')
dbscan.fit(X)

# Applying dbscan to the mall dataset
y_dbscan = dbscan.fit_predict(X)
```

- Some case we should do feature scaling when use euclidean distance
## DBScan in R
- Install 'fpc' package to do dbscaling
- We can do feature scaling because the distance metric is 'euclidean'

```
# Using "fpc" library to clustering with dbscan
# install.packages("fpc")
library(fpc)
set.seed(123)
clusters <- fpc::dbscan(X, eps = 5, MinPts = 5)
```

# Section 23: Association Rule Learning
- Some Association Rule Learning models:
    - Apriori
    - Eclat

# Section 24: Apriori
## Apriori intuition
- Apriori: used to being used in recommendation system
- Apriori thường được áp dụng trong bài toán khai thác tập phổ biến (frequent itemset)
- [Refs](https://ongxuanhong.wordpress.com/2015/08/23/khai-thac-luat-tap-pho-bien-frequent-itemsets-voi-thuat-toan-apriori/)
- Some Apriori keywords:
    - item: A = apple, B = bread, C = cereal, D = donuts, E = eggs
    - itemset: \{A, B, C, D, E\}, \{A, C\}, \{D\}, \{B, C, D, E\}, \{B, C, E\}
    - transaction (TID)
    - frequent item
    - k-itemset:
        - 1-itemset: \{A, B, C\}
        - 2-itemset: \{\{A, B\}, \{A, C\}\}
        - 3-itemset: \{\{A, B, C\}, \{B, C, E\}\}
    - Độ phổ biến (support): `supp(X) = count(X) / |D|. X` trong đó `{B, C}` là tập các hạng mục, D là cơ sở dữ liệu giao dịch
    - Tập phổ biến (frequent itemset): là tập các hạng mục S (itemset) thỏa mãn độ phổ biến tối thiểu (minsupp – do người dùng xác định như 40% hoặc xuất hiện 5 lần). Nếu supp(S) >= minsupp thì S là tập phổ biến
    - Tập phổ biến tối đại (max pattern) thỏa mãn:
        - supp(X) >= minsupp
        - không tồn tại |X’| > |X|, với X’ cũng phổ biến
    - Tập phổ biến đóng (closed pattern) thỏa mãn
        - supp(S) >= minsupp
        - không tồn tại |X’| > |X| mà supp(X’) = supp(X)
    - Luật kết hợp (association rule): kí hiệu `X -> Y`, nghĩa là khi X có mặt thì Y cũng có mặt (với xác suất nào đó). Ví dụ, `A -> B`; `A,B -> C`; `B,D -> E`
    - Độ tin cậy (confidence): được tính bằng `conf(X) = supp(X+Y) / supp(X)`

```
support(M) = (# itemsets which contain M) / (# total itemsets)

confidence(M1 -> M2) = (#itemsets which contain M1 and M2) / (##itemsets which contain M1)

lift(M1 -> M2) = confidence(M1 -> M2) / support(M2)
```

- Apriori steps
    - Step 1: Set a minimum support and confidence
    - Step 2: Take all the subsets in transactions having higher support than minimum support
    - Step 3: Take all the rules of these subsets having higher confidence than minimum confidence
    - Step 4: Short the rules by decreasing lift
## Apriori in Python
- We import `apyori` module to implement apriori algorithm in Python
- Python apriori parameters:
    - min_support
    - min_confidence
    - min_lift
    - min_length

```
# Training Apriori on the dataset
from apyori import apriori
# get items were bought at least 3 times a day
rules = apriori(transactions, min_support = 3*7/7500, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
```

## Apriori in R
- Install `arules` package for using in mining association rules and frequent itemsets
- We can read `transactions` with read.transaction after selecting `arules` library
- Use `itemFrequencyPlot` to show the most frequently bought item

```
# Data preprocessing
# install.packages('arules')
library(arules)

# dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates =TRUE)
summary(dataset)

itemFrequencyPlot(dataset, topN = 20)

# Training Apriori on the dataset
# Support: Buy 3 item for 7 day on week per total 7500 transactions
rules = apriori(data = dataset, parameter = list(support = 3*7/7500, confidence = 0.4))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
```

# Section 25: Eclat
## Eclat intuition
- Support: `support(M) = (# itemsets which contain M) / (# total itemsets)`
- Eclat algorithm
    - Step 1: Set a minimum support
    - Step 2: Take all the subsets in transactions having the higher support than minimum support
    - Step 3: Sort the subsets by decreasing support
- Eclat dont care about `confidence` and `lift` -> __Eclat ~~ Apriori simplified version__

## Eclat in R
- Install `arules` package for using in mining association rules and frequent itemsets
- We can read `transactions` with read.transaction after selecting `arules` library
- Use `itemFrequencyPlot` to show the most frequently bought item

```
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates =TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 20)

# Training Apriori on the dataset
# Support: Buy 3 at least items in one day of 7 days on week per total 7500 transactions
rules = eclat(data = dataset, parameter = list(support = 3*7/7500, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
```

# Section 26: Reinforcement Learning
- Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to solve interacting problems where the data observed up to time `t` is considered to decide which action to take at time `t + 1`.
- Reinforcement Learning is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.
- For example: Reinforcement Learning used to train robot dog walk
- Some Reinforcement Learning models:
    - Upper Confidence Bound (UCB)
    - Thompson Sampling

# Section 27: Upper Confidence Bound (UCB)
## The multi-armed bandit problem
- One armed bandit ~ slot machine (It called bandit because it's the quickest machine that get our money in casino)
- The multi-armed bandit problem (sometimes called the K or N-armed bandit problem) is a problem in which a gambler at a row of slot machines (sometimes known as "one-armed bandits") has to decide which machines to play, how many times to play each machine and in which order to play them -> __We have to find the distribution of these machine to find best machine to exploid__
- Example: Coca-cola provide some adds and let people choose their best ads. They have to test thousand of people (A/B test) to find the distribution of these ads.

## Upper confidence bound intuition (UCB)
- [link1](http://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)
- [link2](https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/)
- UCB algorithm
    - Step 1: At each round n, we consider two numbers for each ad i:
        - Ni(n) - the number of times the ad i was selected up to round n
        - Ri(n) - the sum of rewards of the ad i up to round n
    - Step 2: From these two numbers we compute
        - The average reward of ad i up to round n: `ri(n) = Ri(n) / Ni(n)`
        - The confidence interval `[ri(n) - delta(n), ri(n) + delta(n)]` around n with `delta = sqrt(3*log(N)/(2*Ni(n)))`
    - Step 3: We selected the ad i that has the maximum UCB `ri(n) + delta(n)`
- There's no library for R and Python of UCB algorithm so we need to self implement the algorithm

# Section 28: Thompson sampling
## Thompson sampling intuition
- Thompson sampling algorithm
    - Step 1: At each round *n*, we consider two number for each ad *i*:
        - N1_i(n) - the number of times the ad *i* got reward 1 up to round *n*,
        - N0_i(n) - the number of times the ad *i* got reward 0 up to round *n*
    - Step 2: for each ad *i*, we take a random draw from the distribution below:
        -   `theta_i(n) = beta(N1_i(n) + 1, N0_i(n) + 1)`
    - Step 3: We select the ad that has the highest `theta_i`
- [Readmore Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- [Beta distribution vs normal distribution](https://www.quora.com/What-are-the-key-differences-between-normal-distribution-and-beta-distribution)

## Comparison between UCB vs Thompson sampling

|UCB|Thompson sampling|
|---|---|
|Deterministic|Probabilistic|
|Require update at every round (better because update every round. But more heavy)|Can accommodate delayed feedback (Good performance for large dataset)|
||Better empirical evidence|

# Section 29: Natural language processing
- NLP is a branch of ML
- Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.
- You can also use NLP on a text review to predict if the review is a good one or a bad one. You can use NLP on an article to predict some categories of the articles you are trying to segment. You can use NLP on a book to predict the genre of the book. And it can go further, you can use NLP to build a machine translator or a speech recognition system, and in that last example you use classification algorithms to classify language. Speaking of classification algorithms, most of NLP algorithms are classification models, and they include Logistic Regression, Naive Bayes, CART which is a model based on decision trees, Maximum Entropy again related to Decision Trees, Hidden Markov Models which are models based on Markov processes.
- A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

## NLP in python (Bag of words model)
- Step 1: Load the dataset
- Step 2: Cleaning the texts. Use `re` and `nltk` library
    - Remove numbers
    - Remove punctuation
    - Change text into lowercase
    - Remove stop words. Import the `stopwords` from `nltk.corpus`
    - Do stemming. Ex: `loved` -> `love`
    - Joining back the string

```
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
```

- Step 3: Creating the Bag of Words model
    - We tokenization the words
    - Each word will be an independence variables
    - Use `countVectorizer` in `sklearn.feature_extraction.text` library to tokenizing
    - `countVectorizer` also have cleaning text feature but we should do it manually for support more options and more controls
    - `countVectorizer` parameters:
        - max_features: use most frequent words

```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
```

- Step 4: classification
    - we can choose to best options for NLP: __decision tree (or random forest) and naive bayes__

## NLP in R (Bag of words model)
- Step 1: Load the dataset. Use `read.delim()` for tabular csv file
```
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
```

- Step 2: Cleaning the texts. Install `tm` package to `cleaning the texts`
    - Change text into lowercase
    - Remove numbers
    - Remove punctuation
    - Remove stop words. Install `SnowballC` package to use `stopwords` function
    - Do stemming. Ex: `loved` -> `love`
    - Remove extra spaces

```
# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
```

- Step 3: Creating the Bag of Words model
    - We tokenization the words
    - Each word will be an independence variables
    - Use `DocumentTermMatrix` function to tokenize corpus

```
# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
```

- Step 4: classification
    - we can choose to best options for NLP: __decision tree (or random forest) and naive bayes__

# Section 30: Deep Learning
- Deep Learning models can be used for a variety of complex tasks:
    - Artificial Neural Networks for Regression and Classification
    - Convolutional Neural Networks for Computer Vision
    - Recurrent Neural Networks for Time Series Analysis
    - Self Organizing Maps for Feature Extraction
    - Deep Boltzmann Machines for Recommendation Systems
    - Auto Encoders for Recommendation Systems
- Some basic deep learning model
    - Artificial Neural Networks for a Business Problem
    - Convolutional Neural Networks for a Computer Vision task

## What is deep learning
- Geoffrey Hilton: God father of `Deep learning`
- Deep learning model mimics human brain with a lots of layers:
    - Input layer
    - Lots of hidden layers
    - And output

# Section 31: Artificial Neural Networks
## The neuron
- Neuron + dendrites + axon
    - Dendrites: receiver for the signal
    - Axon: transmitter for the signal
- Basically the neuron get signal from other neurons

```
Input signal 1 --
                  >
Input signal 2 ---> Neuron -> Output
      ...         >
Input signal m --
```

##The activation function
- Threshold function

```
θ(x) = 1 if x≥0
       0 if x<0
```

- Sigmoid function

```
θ(x) = 1 / (1 + e^(-x))
```

- Rectifier function (King of functions, one of the most functions in Neural network)
```
θ(x) = max(x,0)
```
- Hyperbolic tangent (tanh)
```
θ(x) = (1-e^(-2x)) / (1+e^(-2x))
```

- Assuming the dependent variables is binary (0 or 1) we can choose:
    - threshold activation function
    - sigmoid activation function
    - Data-scientist usually use `rectifier` function for hidden layers and `sigmoid` function for out put layer

## How do the NNs learn?
```
X1 --
     >
X2 ---> ∑(w_i * x_i) -> y^
 ... >
Xm --
```

- Given a initial values of weight (W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>, ... W<sub>m</sub>)
- The NN compute the output values y^ and the cost value: `C = 0.5 * ∑(y^ - y) ^ 2`
- Then, the NN adjusts W<sub>1</sub>, W<sub>2</sub>, ... W_m to minimize the cost value `C`
- When the `C` small enough we have a new NN model

## Gradient descent
- `Gradient descent` is a method for adjust the weights of the NNs.
- There's a method, which used a brute force way to try thousand of weights value and compute the cost function. The weight with smallest value of cost function make our models --> this method is time consuming.
- `Gradient descent` has a more effectively and quickly method for find minimum cost value using `derivative`

## Stochastic gradient descent
- `Stochastic gradient descent` (or `mini batch gradient descent`) is like `gradient descent` (or `batch gradient descent`) but we do with every steps of added values. The `gradient descent` use whole data.
- This method help us avoid local minimum problem in `batch gradient descent`.

## Back propagation
- `Back propagation` is an advance algorithm allows us to adjust weights after calculate error (cost)

## Install Theano, Tensorflow, Keras
- `Theano` opensource numerical computation library, can run in GPU. Developed by group of people in the University of Montreal
```
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

- `Tensorflow` opensource numerical computation library, can run in GPU. Developed originally by Google but now it is a Apache project. There's some ways to install `tensorflow`
  - From a docker
  - From a virtual machine
  - Anaconda intergrated version. Follow [link](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)

```
Step 1: Install anaconda
Step 2: Create a conda environment named tensorflow
  conda create -n tensorflow
Step 3: Activate the conda environment
  source activate tensorflow
Step 4: Issue a command of the following format to install TensorFlow inside your conda environment:
  pip install --ignore-installed --upgrade TF_PYTHON_URL
  pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
```

- `Keras` is an amazing library to build deep learning model. Build by a machine learning scientist at Google.

```
pip install --upgrade keras
```

## ANN in python
- Use classification template for deep learning classification because deep learning model is also a classification model
- Do data pre-processing like we do in other model
- We use `keras` in our example. By default, our `keras` using `Tensorflow` backend. if we need use `keras` with `Theano`, we need to manually config `keras`.
    - Use `Sequential` model to initialize our neural network
    - Use `Dense` model to build layers of our ANN

```
import keras
from keras.models import Sequential
from keras.layers import Dense
```

- Tips: By experiments `the number of nodes in hidden layers equal the average between input layer and output layer - Udemy Teacher`
- If the output layer has more than 2 layers we should use the activation function is `softmax` instead of `sigmoid`

```
# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, # Tips: choose node numbers of hidden layers equal average of node in both intput and output layers
                                     # Input layers: 11 nodes - 11 independent variables
                                     # Output layers: 1 node - 1 dependent variable
                     init = 'uniform',
                     activation = 'relu', # rectifier activation function for hidden layer
                     input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, # Tips: choose node numbers of hidden layers equal average of node in both intput and output layers
                                     # Input layers: 11 nodes - 11 independent variables
                                     # Output layers: 1 node - 1 dependent variable
                     init = 'uniform',
                     activation = 'relu')) # rectifier activation function for hidden layer

# Adding the output layer
classifier.add(Dense(output_dim = 1, # We have only 1 dependent variable
                     init = 'uniform',
                     activation = 'sigmoid')) # Use sigmoid function for output layer

# Copy the ANN
classifier.compile(optimizer = 'adam', # Optimizer is algorithm used to find initialized set-up weights
                                       # 'adam' is one of the effective stochastic gradient descend
                   loss = 'binary_crossentropy', # loss function equivalent with sigmoid activation function
                   metrics=['accuracy']
                   )
```

- When fitting the ANN with the training set using we need to specify 2 more parameters
    - batch_size (by default, the value is 32. We should choose smaller value. Maybe 10)
    - nb_epoch: number of epoch

```
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
```

## ANN in R
- Use classification template for deep learning classification because deep learning model is also a classification model
- Using `h2o` package (the best package to build deep learning model in R)
    - It's opensource. Allow connect an instance of a computer system
    - Offers a lot of options to build deep learning modal

```
install.packages('h2o')
```

- Building ANN classifier with `h2o` with simple commands below.
  - When use `h2o` library, we need to convert dataframe to h2o dataframe like `as.h2o(training_set)`

```
# Fitting ANN to the Traning set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set), # convert dataframe to h2o training frame
                              activation = 'Rectifier',
                              hidden = c(2,6), # 2 hidden layers and each layer has 6 nodes
                              epochs = 100,
                              train_samples_pser_iteration = -2) # batch size will be auto tunned
```

# Section 32: Convolutional neural network
- CNN is an ANN and a great deeplearning model for computer vision (to classify images, photographs, videos...)

## Convolutional neural network
- Convolutional = `tích chập`. Refs links:
  - [link](https://www.stdio.vn/articles/read/386/phep-tich-chap-trong-xu-ly-anh-convolution)
  - [youtube](https://www.youtube.com/watch?v=gFELyrIx010)
- Usually using in image classification, use on car driving...
- `Yann lecun` god father of `convolutional neural network`

```
Input image --> CNN --> Output label (image class)
```

## Step 1: Convolution function
- Input image (as matrix of pixels) * Feature detector (kernel matrix) ==> Feature map
- Ma trận kernel trượt qua ma trận ảnh -> tính ra ma trận tích chập. Sử dụng feature map cho phép tìm ra trong input image vùng thông tin có giá trị trùng lặp lớn nhất so với feature mask. Đó là cơ sở chính cho các thuật toán nhận dạng (ví dụ face detection trong ảnh facebook...). Ví dụ: vùng giống `feature detector` nhất cho giá trị lớn nhất trên `feature map`

```
Input image               Feature detector      Feature map

0 0 0 0 0 0 0             0 0 1                 0 1 0 0 0

0 1 0 0 0 1 0             1 0 0                 0 1 1 1 0

0 0 0 0 0 0 0     (X)     0 1 1       (=)       1 0 1 2 1
 |-----|                                         |-|
0|0 0 1|0 0 0                                   1|4|2 1 0
 |     |                                         |-|
0|1 0 0|0 1 0                                   0 0 1 2 1
 |     |
0|0 1 1|0 0
 |-----|
0 0 0 0 0 0 0
```

- In convolutional neural network -> we create feature detectors to get multiple feature maps
- We can use some specific of feature detectors to apply into image for bluring, sharpening ... For example
  - Bluring with
  ```
  0  0  0  0  0
  0  1  1  1  0
  0  1  1  1  0
  0  1  1  1  0
  0  0  0  0  0
  ```
  - Sharpening
  ```
  0  0  0  0  0
  0  0 -1  0  0
  0 -1  5 -1  0
  0  0 -1  0  0
  0  0  0  0  0
  ```
  - Edge enhance
  ```
  0  0  0  0  0
  0  0  0  0  0
  0 -1  1  0  0
  0  0  0  0  0
  0  0  0  0  0
  ```
  - Edge detect ==> very importance feature for us to detect objects
  ```
  0  0  0  0  0
  0  0  1  0  0
  0  1 -4  1  0
  0  0  1  0  0
  0  0  0  0  0
  ```
  - Emboss
  ```
  0  0  0  0  0
  0 -2 -1  0  0
  0 -1  1  1  0
  0  0  1  2  0
  0  0  0  0  0
  ```
  - `Teacher favorite filter -> it creates cartoon effect :D`
  ```
  0  0  0  0  0
  0  1  0 -1  0
  0  2  0 -2  0
  0  1  0 -1  0
  0  0  0  0  0
  ```

## Step 1b: ReLU layers
- Relu layer help us filter values. For example, we filter all black pixels from image

## Step 2: Pooling
- Pooling -> __reduce number of parameters__
- Some type of pooling:
  - max pooling
  - mean pooling
  - average pooling
  - min pooling
- Pooling được áp dụng vào feature maps. Pooling đảm bảo khi hình ảnh xoay, nghiêng, thu hẹp (nói chung là nhiều hình dạng của ảnh) đểu có thể attract ra được đặc trưng cơ bản của feature map)
- [Example for pooling visualisation](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)
- [link](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf)

## Step 3: Flattening
- `Matrix` to `Array`. For example
```
1 1 0
4 2 1    -->   1 1 0 4 2 1 0 2 1
0 2 1
```
- After flattening we apply it as the input layer of a ANN

## Step 4: Full connection
- After convolutioning -> pooling -> flattening -> we use ANN to classify objects

## Summary
```
             convolutional                         pooling                     flattening
Input image       -->       convolutional layers     -->      pooling layer       -->     input layer of a future ANN
             ReLU rectifier
```

## Softmax & Cross entropy
- Softmax function: ___f<sub>j</sub>(z) = e<sup>z</sup><sub>j</sub> / ∑<sub>k</sub> e<sup>z</sup><sub>k</sub>___
- Cross entropy function: like `mean squared error` function in ANN, cross entropy is a way to compute loss (and maybe a loss function for CNN)
- For example when compare two NN with value like belows
  - NN1

| Row | Dog<sup>^</sup> | Cat<sup>^</sup> | Dog | Cat |
| --- |:----:|:----:|:---:|:---:|
| #1  | 0.9  | 0.1  | 1   | 0   |
| #2  | 0.1  | 0.9  | 0   | 1   |
| #3  | 0.4  | 0.6  | 1   | 0   |

  - NN2

| Row | Dog<sup>^</sup> | Cat<sup>^</sup> | Dog | Cat |
| --- |:----:|:----:|:---:|:---:|
| #1  | 0.6  | 0.4  | 1   | 0   |
| #2  | 0.3  | 0.7  | 0   | 1   |
| #3  | 0.1  | 0.9  | 1   | 0   |

  - Classification error: NN1 = NN2 = 0.33
  - Mean squared error: NN1 = 0.25; NN2 = 0.71
  - __Cross entropy: NN1 = 0.38; NN2 = 1.06__
  - __Cross entropy__ is more effectively than __mean squared error__ function

## Convolutional Neural Network in Python
- Train CNN to predict photo of dogs or cats
- [The dataset for this example](http://www.superdatascience.com/wp-content/uploads/2017/03/Convolutional-Neural-Networks.zip)
- Prepare data: separate cats and dogs pictures data in folder -> `keras` will automatically know and label our input
- CNN python initialising steps:
  - Step 1: Convolution
```
classifier.add(Convolution2D(
        32, # number of feature detectors
            # We can use larger number when working with GPU
        3, # row of each feature detector
        3, # columns of reach feature detector
        input_shape=(64,64,3),  # if using Theano backend input_shape=(3,64,64)
                                # We can use larger shape size when working with GPU. Ex: 256 * 256
                                # 3 mean we working with color images. These are R,G,B channels
        activation = 'relu' # Use rectifier activation function -> make sure we working with non negative values
        ))
```
  - Step 2: Max pooling
```
# Do Pooling to reduce number of feature maps
classifier.add(MaxPooling2D(pool_size = (2,2))) # Larger pool size -> more information we loose
```
  - Step 3: Flattening
```
classifier.add(Flatten()) # Keras automatically know to flatten previous layer
```
  - Step 4: Full connection
```
classifier.add(Dense(output_dim = 128, # By experiment, we should choose number of nodes around 100
                     activation = 'relu')) # Use rectifier activation function
classifier.add(Dense(output_dim = 1, # By experiment, we should choose number of nodes around 100
                     activation = 'sigmoid')) # Use sigmoid function because we have 1 output
```
- [Keras image preprocessing](https://keras.io/preprocessing/image/)
- Fitting the CNN to the images
```
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set', # change to your real traning image folder
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') # We have two class CAT and DOG

test_set = test_datagen.flow_from_directory(
        'dataset/test_set', # change to your real traning image folder
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)
```

# Section 32: CNN in R
- Deep Water is a library in R that support CNN but it still in early stage.
- R is an amazing programming language for highly advanced statistics, but when it comes to Deep Learning for Computer Vision, Python is more mature
- Refs:
    - [h2o](http://www.h2o.ai/)
    - [Deep Water](http://www.h2o.ai/deep-water/)
    - [CNN with h2o](https://community.h2o.ai/questions/452/rnncnn-with-h2o.html)
    - [Deep Water in GitHub](https://github.com/h2oai/deepwater)

# Section 33: Dimensionality Reduction
- There are two types of Dimensionality Reduction techniques:
    - Feature Selection
    - Feature Extraction
- Feature Selection techniques are Backward Elimination, Forward Selection, Bidirectional Elimination, Score Comparison and more
- Feature Extraction techniques:
    - Principal Component Analysis (PCA)
    - Linear Discriminant Analysis (LDA)
    - Kernel PCA
    - Quadratic Discriminant Analysis (QDA)

# Section 34: Principal Component Analysis
- From the `m` independent variables of the dataset, PCA extracts `p ≤ m` new independent variables that explain the most variance of the dataset, __regardless of the dependent variable__
    - `Variance`: In probability theory and statistics, variance is the expectation of the squared deviation of a random variable from its mean
    - The new `p` independent variables are not in the original independent variables. It's the main different of `Feature selection` and `Feature extraction`
- PCA extracts the top principal components (independent variables). Top principal variables has the maximum `variance`
- `In probability theory and statistics, variance is the expectation of the squared deviation of a random variable from its mean`
- After extracts these top principal components, we can easily visualize the results
- Ref:
  - [Link](http://sebastianraschka.com/Articles/2014_python_lda.html)

## PCA in python
- Use `PCA` library from `sklearn.decomposition` packages to apply feature reduction
```
# Applying PCA
# Snippest for show explained variance to get the top principal variables
"""from sklearn.decomposition import PCA
pca = PCA(n_components = None)  # to see the cumulative varient curve
                                # We can specified our value: such as 2
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_"""

from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # We select 2 variables after investigate the `explained_variance`
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
```

## PCA in R
- Install `caret` library to do feature extraction in R
```
install.packages('caret')
library(caret)

pca = preProcess(x = training_set[-14],
                 method = 'pca',
                 pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2,3,1)] # Move the dependent variable to the last column
```

# Section 35: Linear Discriminant Analysis (LDA)
- From the `n` independent variables of your dataset, LDA extracts `p ≤ n` new independent variables that separate the most the classes of the dependent variable.

## LDA in python
- Use `LinearDiscriminantAnalysis` library from `sklearn.discriminant_analysis` to apply LDA
```
# Applying LDA
# Snippest for show explained variance to get the top principal variables
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_train = lda.fit_transform(X_train, y_train) # LDA is a supervise model -> we need `fit_transform` with both X_train and y_train
x_test = lda.transform(X_test)
```

## LDA in R
- Install `MASS` library to do feature extraction in R
```
# Applying LDA
# install.packages('MASS')
library(MASS)
lda = lda(formula = Customer_Segment ~ .,
          data = training_set)
# We have 3 classes in Customer_Segment -> we got 3-1 = 2 discriminent
# LDA required output data as a matrix
training_set = as.data.frame(predict(lda, training_set))
training_set = training_set[c(5,6,1)]
test_set = as.data.frame(predict(lda, test_set))
test_set = test_set[c(5,6,1)]
```

# Section 36: Kernel PCA
- Kernel PCA is a `kernelization version of PCA`
- Kernel PCA use kernel trick to convert data from original form to new data in a higher dimension to avoid too highly compute intensive computation.

## Kernel PCA in python
- Use `KernelPCA` library from `sklearn.decomposition` to apply Kernel PCA
- Use `KernelPCA` like `PCA` but we put one more feature called `kernel`. Use this parameter like we used in `Kernel SVM`
```
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
k_pca = KernelPCA(n_components = 2, # We select 2 variables after investigate the `explained_variance`
                  kernel = 'rbf') # We can use the kernel same like we did with Kernel SVM
X_train = k_pca.fit_transform(X_train)
X_test = k_pca.transform(X_test)
```

## Kernel PCA in R
- install `kernlab` to implement `kernel PCA` in R
- Specify the kernel function like we did in `svm`
```
# install.packages('kernlab')

library(kernlab)
k_pca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(k_pca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(k_pca, test_set))
test_set_pca$Purchased = test_set$Purchased
```

# Section 37: Model Selection & Boosting
- Some questions after building our model are:
  - How to deal with the bias variance tradeoff when building a model and evaluating its performance ?
  - How to choose the optimal values for the hyperparameters (the parameters that are not learned) ?
  - How to find the most appropriate Machine Learning model for my business problem ?
- There's some techniques that do the `Model selection`:
  - k-Fold Cross Validation
  - Grid Search

# Section 38: Model Selection
## k-Fold Cross Validation in Python
- k-Fold split the training set to 10 folds. Train the model in the 9 folds and test in the last remaining fold -> There will be 10 combinations of training and testing folds.
- The accuracy we get for our model can be counted as the `mean` value of the accuracies we get after doing k-Fold
- Using `cross_val_score` library from `sklearn.model_selection` to apply `k-Fold`
```
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10) # Number of folds
accuracies.mean()
accuracies.std()
```

## k-Fold Cross Validation in R
- The accuracy we get for our model can be counted as the `mean` value of the accuracies we get after doing k-Fold
- Install `caret` package to use the `lapply` function and implement `k-Fold Cross Validation`. The `lapply` ~ `list apply`
```
# install.packages('caret')

library(caret)
folds = createFolds(training_set$Purchased,
                    k = 10) # Number of folds
# lapply: list apply
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]

  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')

  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,2] + cm[2,1] + cm[1,1] + cm[2,2])
  return(accuracy)
})
mean(as.numeric(cv))
```

## Grid search in Python
- How do I know which model to choose? Which model is the best one? ==> `Grid search` help us to find the best parameters -> `optimal parameters`
- Using `GridSearchCV` library from `sklearn.model_selection` to implement `Grid search`
- We have to build a dictionary of parameters and pass it as `param_grid` parameter of `GridSearchCV` function
```
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.5, 0.1, 0.01, 0.001]},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4, 5]}
             ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy', # specify the performance metric
                           cv = 10 # Number of folds (similar to k-Fold)
                           )
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
```

## Grid search in R
- Install `caret` package to implement `Grid search` in R
- For parameters tuning -> we should build `caret` package
```
# Applying Grid Search to find the best parameters
# install.packages('caret')
library(caret)
classifier = train(form = Purchased ~ .,
                   data = training_set,
                   method = 'svmRadial' # Avaiable methods: https://topepo.github.io/caret/available-models.html
                   )
classifier$bestTune
```

# Section 39: XGBoots
- One of the most popular algorithm in machine-learning
- XGBoots has 3 quality: high performance, fast execution speed, we can keep all the interpretation of our problem
- No need to implement `feature scaling` in `XGBoots`
- [Install XGBoots](http://xgboost.readthedocs.io/en/latest/build.html)
