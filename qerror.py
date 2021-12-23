import pandas as pd
import numpy as np
from numpy.random import seed
seed(1)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


iris =load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

df['Species']=pd.Categorical.from_codes(iris.target,iris.target_names)
print(df["Species"].head())
print(df["Species"].value_counts())


print(df)
#select features
x=df.iloc[:,0:4]

kmeans=KMeans()

wcss= []
for i in range(1,7):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
#print(wcss)
print(kmeans.inertia_)


# The Elbow method
import matplotlib.pyplot as plt
number_clusters=range(1,7)
plt.plot(number_clusters,wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum Of Squares")
plt.show()

