# Title     : Eclat
# Created by: trankhai
# Created on: 10/6/17

# Data preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates =TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 20)

# Training Apriori on the dataset
# Support: Buy 3 at least items in one day of 7 days on week per total 7500 transactions
rules = eclat(data = dataset, parameter = list(support = 3*7/7500, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])