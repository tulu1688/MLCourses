# K-Nearest neighbors for Iris dataset

# Refs: https://machinelearningcoban.com/2017/01/08/knn/

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Import the iris dataset in sklearn.datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Importing the dataset from iris.csv
"""
dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the K-NN classifier to Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = 'minkowski', # metric: minkowski or euclidean
                                  p = 2) # norm 2 (euclid); p=1 => norm 1 (tri tuyet doi)
classifier.fit(X_train, y_train)

# Predict the Test set result
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10) # Number of folds
accuracies.mean()
accuracies.std()