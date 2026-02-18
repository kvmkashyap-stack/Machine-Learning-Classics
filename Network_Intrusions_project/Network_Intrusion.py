import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

df = pd.read_csv("net.csv")
df = df.select_dtypes(include=[np.number])
df = df.dropna()

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_train = kmeans.fit_predict(X_train_pca)
kmeans_test = kmeans.fit_predict(X_test_pca)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_train = dbscan.fit_predict(X_train_pca)
dbscan_test = dbscan.fit_predict(X_test_pca)

plt.figure()
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=kmeans_train)
plt.title("KMeans Train Clusters")
plt.show()

plt.figure()
plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c=kmeans_test)
plt.title("KMeans Test Clusters")
plt.show()

plt.figure()
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=dbscan_train)
plt.title("DBSCAN Train")
plt.show()

plt.figure()
plt.scatter(X_test_pca[:,0], X_test_pca[:,1], c=dbscan_test)
plt.title("DBSCAN Test")
plt.show()




print(pd.Series(dbscan_train).value_counts())
print(pd.Series(dbscan_test).value_counts())
