# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# Fitting SVR to the dataset
# install.packages('e1071')
library('e1071')
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# Fitting Polynomial Linear Regression to the dataset
poly_reg = lm(formula = Salary ~ .,
                data = dataset)

# Predict a new result with Regression
predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression modal with smooth curve
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), color = 'blue') +
    ggtitle('Title') +
    xlab('X label') +
    ylab('Y label')