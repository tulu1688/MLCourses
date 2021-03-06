# K-means clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withins)
plot(1:10, wcss, type = 'b', main = paste('Clusters of clients'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Applying k-means to the mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualising the dbscan clusters with "factoextra"
# install.packages('factoextra')
library(factoextra)
fviz_cluster(kmeans, data = X, stand = FALSE,
ellipse = FALSE, show.clust.cent = FALSE,
geom = "point",palette = "jco", ggtheme = theme_classic())

# Visualising the clusters with clusplot
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Cluster of the clients"),
         xlab = "Annual income",
         ylab = "Spending score")