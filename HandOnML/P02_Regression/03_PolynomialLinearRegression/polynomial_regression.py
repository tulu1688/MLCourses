# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X must be matrix not a vector. So we need to specify the range of collumn [1:2]
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# The dataset is too short. So we donot need to separate the dataset

# Feature Scaling
# Polynomial feature scaling libray do consist feature scaling. No need to process feature scaling

# Fitting the Linear regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting the Polynomial linear regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression Result
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Linear Regression Result
X_grid = np.arange(min(X),max(X),0.1)
# Change vector to matrix
X_grid = X_grid.reshape((len(X_grid)), 1)
# Draw prediction line with X_grid
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()
