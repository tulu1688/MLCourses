# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Regression to the dataset
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], # dataframe
                         y = dataset$Salary, # Vector
                         ntree = 500) # Number of trees to grow

# Predict a new result with Regression
predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression modal with smooth curve
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
x_grid_frame = data.frame(Level = x_grid,
            Level2 = x_grid^2,
            Level3 = x_grid^3,
            Level4 = x_grid^4)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = x_grid_frame)), color = 'blue') +
    ggtitle('Truth or Bluff (Random Forest Regression)') +
    xlab('Position Level') +
    ylab('Salary')