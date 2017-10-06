# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501): # exclude uper bound 7501
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) # exclude index = 20

# Training Apriori on the dataset
