# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[,4:14]

# Encoding the categorical variables as factors
# For some deep learning package, it only working with numerical variables -> Convert 
#   categorical variable to numeric variables
dataset$Geography = as.numeric(factor(dataset$Geography,
                              levels = c('France', 'Spain', 'Germany'),
                              labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                              levels = c('Female', 'Male'),
                              labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# For colum that categorical trasnformed by factor function, these are not numeric so can't be scale
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Traning set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set), # convert dataframe to h2o training frame
                              activation = 'Rectifier',
                              hidden = c(2,6), # 2 hidden layers and each layer has 6 nodes
                              epochs = 100,
                              train_samples_per_iteration = -2) # batch size will be auto tunned

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the confusion matrix
cm = table(test_set[,11], y_pred)

# Disconnnecting from h2o servers
h2o.shutdown()