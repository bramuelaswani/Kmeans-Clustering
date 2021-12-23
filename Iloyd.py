import pandas as pd
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np 
from numpy.random import seed
seed(1)

# Lets create our random data
random_test_data=make_blobs(n_samples=400,n_features=2,centers=4,random_state=7,cluster_std=1.6)
#print(random_test_data)

data_points=random_test_data[0]


#save the random data as a txt file
np.savetxt("data1.txt",data_points,fmt="%s", delimiter=",")


dataset=pd.read_csv(input("Enter the data file here: "))
k=int(input("Enter k value here:"))
r=int(input("Enter r value here:"))

#print(dataset,k,r)
kmeans=KMeans(k)

wcss= []
for i in range(1,r):
    kmeans=KMeans(i)
    kmeans.fit(dataset)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
#print(wcss)
print("Quantization Error:"+ str(kmeans.inertia_))

kmeans=KMeans(k)
identified_clusters=kmeans.fit_predict(dataset)
print(identified_clusters)

