# Title     : Apriori
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
# Support: Buy 3 item for 7 day on week per total 7500 transactions
rules = apriori(data = dataset, parameter = list(support = 3*7/7500, confidence = 0.4))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])