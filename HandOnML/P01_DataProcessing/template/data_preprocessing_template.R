# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# Takecaring of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Transform categorical column
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Germany', 'Spain'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes','No'),
                           labels = c(1,0))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# For colum that categorical trasnformed by factor function, these are not numeric so can't be scale
# example code: training_set = scale(training_set)
training_set[,2:3] = scale(training_set[,2:3])
# example code: test_set = scale(test_set)
test_set[,2:3] = scale(test_set[,2:3])