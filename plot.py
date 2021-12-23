import pandas as pd
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np 
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt

# Lets create our random data
random_test_data=make_blobs(n_samples=400,n_features=4)
print(random_test_data)

plt.scatter(random_test_data[0][:,0],random_test_data[0][:,1])
plt.title("Random_test_data Before clustering")
plt.show()

data_points=random_test_data[0]


#save the random data as a txt file
np.savetxt("data1.txt",data_points,fmt="%s", delimiter=",")


dataset=pd.read_csv("data1.txt")
k=3
r=8

#print(dataset,k,r)
kmeans=KMeans(k)

wcss= []
for i in range(1,r):
    kmeans=KMeans(i)
    kmeans.fit(dataset)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
#print(wcss)
# The Elbow method
import matplotlib.pyplot as plt
number_clusters=range(1,8)
plt.plot(number_clusters,wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum Of Squares")
plt.show()

#print("Quantization Error:"+ str(kmeans.inertia_))

kmeans=KMeans(3)
identified_clusters=kmeans.fit_predict(dataset)
#print(identified_clusters)
plt.scatter(data_points[identified_clusters==0,0],data_points[identified_clusters==0,1],s=60,color="red")
plt.scatter(data_points[identified_clusters==1,0],data_points[identified_clusters==1,1],s=60,color="green")
plt.scatter(data_points[identified_clusters==2,0],data_points[identified_clusters==2,1], s=60, color="orange")
plt.scatter(data_points[identified_clusters==3,0],data_points[identified_clusters==3,1],s=60, color="yellow")
plt.show()