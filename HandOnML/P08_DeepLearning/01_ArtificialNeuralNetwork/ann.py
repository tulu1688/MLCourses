# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Let's make the ANN
# Import the keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

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

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predict the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)