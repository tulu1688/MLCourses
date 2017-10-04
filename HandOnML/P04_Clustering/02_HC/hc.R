# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[3:4]

# Using dendogram to find the optimal number of clusters
dendogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendogram,
    main = paste('Dendogram'),
    xlab = 'Customers',
    ylab = 'Euclidean distances')

# Fitting the hierarchical clustering to mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(X,
        y_hc,
        lines = 0,
        shade = TRUE,
        color = TRUE,
        labels = 2,
        plotchar = FALSE,
        span = TRUE,
        main = paste("Cluster of the customers"),
        xlab = "Annual income",
        ylab = "Spending score")