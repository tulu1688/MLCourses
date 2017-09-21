# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# In R the LinearRegression lib has implement the feature scaling -> no need to manually implement it

# Fitting the simple linear regression to Traning set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predict the Test set
y_pred = predict(regressor, newdata = test_set)

# Prepare visualising library
# install.packages("ggplot2")
library(ggplot2)

# Visualising the Training set result
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), color = 'blue') +
  ggtitle('Salary vs Years of experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualising the Test set result
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), color = 'blue') +
  ggtitle('Salary vs Years of experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')