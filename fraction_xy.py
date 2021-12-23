import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn import cluster
seed(1)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

iris =load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

df['Species']=pd.Categorical.from_codes(iris.target,iris.target_names)
print(df["Species"].head())
print(df["Species"].value_counts())


print(df)
#select features
x=df.iloc[:,0:4]
print(x)

#clustering
kmeans = KMeans(3)
kmeans.fit(x)

#clustering results
identified_clusters=kmeans.fit_predict(x)
print(identified_clusters)

#Create anew dataframe with predictions besides it
df_with_clusters=df.copy()
df_with_clusters["clusters"]=identified_clusters
print(df_with_clusters.head(20))

#Frequencies in cluster labels
print(np.unique(df_with_clusters.clusters,return_counts=True))

plt.scatter(df_with_clusters["sepal length (cm)"],df_with_clusters["petal length (cm)"],c=df_with_clusters["clusters"],cmap="rainbow")
plt.title("sepal length vs petal length")
plt.legend(df_with_clusters["clusters"])
plt.show()

#plot2
# plt.scatter(df_with_clusters["sepal width (cm)"],df_with_clusters["petal width (cm)"],c=df_with_clusters["clusters"],cmap="rainbow")
# plt.title("sepal witdth vs petal width")
# plt.show()

