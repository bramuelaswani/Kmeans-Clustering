from sklearn.cluster import KMeans
import pandas as pd

dataset=pd.read_csv(input("Enter the data file here: "))
k=int(input("Enter k value here:"))
r=int(input("Enter r value here:"))

#print(dataset,k,r)
kmeans=KMeans(k)

wcss= []
for i in range(1,r):
    kmeans=KMeans(n_clusters = 3, init = "k-means++", random_state = 7)
    kmeans.fit(dataset)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
#print(wcss)
print("Quantization Error:"+ str(kmeans.inertia_))

kmeans=KMeans(k)
identified_clusters=kmeans.fit_predict(dataset)
print(identified_clusters)

