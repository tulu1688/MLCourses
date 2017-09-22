# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting multiple linear regression in Traning set
# Profit ~ R.D.Spend + Administration + Marketing.Spend + State = Profit ~ .
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set result
y_pred = predict(regressor, newdata = test_set)

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
