# DBScan clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using "fpc" library to clustering with dbscan
# install.packages("fpc")
library(fpc)
set.seed(123)
clusters <- fpc::dbscan(X, eps = 5, MinPts = 5)

# Visualising the dbscan clusters with "factoextra"
# install.packages('factoextra')
library(factoextra)
fviz_cluster(clusters, data = X, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic())

# Visualising the clusters with clusterplot
library(cluster)
clusplot(X,
         clusters$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste("Cluster of the clients"),
         xlab = "Annual income",
         ylab = "Spending score")