import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

MD_x = data[['Attribute1', 'Attribute2', 'Attribute3', '...']]

scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

 clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
MD_k4 = kmeans.fit_predict(MD_x_scaled) 
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.bar(range(len(MD_x.columns)), MD_x.iloc[MD_k4 == i].mean(), alpha=0.7, label=f'Segment {i+1}')

plt.xticks(range(len(MD_x.columns)), MD_x.columns, rotation=45)
plt.legend()
plt.title('Segment Profile Plot')
plt.show()

pca = PCA(n_components=2)
MD_pca = pca.fit_transform(MD_x_scaled)

plt.figure(figsize=(8, 8))
for i in range(4):
    plt.scatter(MD_pca[MD_k4 == i, 0], MD_pca[MD_k4 == i, 1], label=f'Segment {i+1}')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='o', s=200, alpha=0.7, label='Segment Centers')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Segment Separation Plot')
plt.show()