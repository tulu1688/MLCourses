# K-means clustering

# Ref: https://github.com/tiepvupsu/tiepvupsu.github.io/blob/master/assets/kmeans/Kmeans2.ipynb 

# Importing the libraries
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Importing the dataset
img = mpimg.imread('girl.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
#plt.axis('off')
plt.show()
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

# Splitting the dataset into the Training set and Test set
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the mall dataset
cluster_no = 4 # 3 is the number of cluster after doing elbow method
kmeans = KMeans(n_clusters = cluster_no, init = 'k-means++', max_iter = 300, n_init = 100, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
label = kmeans.predict(X)

# Visualising the clusters
img2 = np.zeros_like(X)
# replace each pixel by its center
for k in range(cluster_no):
    img2[label == k] = kmeans.cluster_centers_[k]

img3 = img2.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(img3, interpolation='nearest')
plt.axis('off')
plt.show()