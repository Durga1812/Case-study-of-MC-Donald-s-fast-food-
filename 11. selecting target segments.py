import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

np.random.seed(1234)
MD_x = pd.DataFrame(np.random.rand(100, 10), columns=['feature_{}'.format(i) for i in range(10)])
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    labels = kmeans.fit_predict(data)
    return labels 

def  calculate_stability(labels1, labels2):
    return adjusted_rand_score(labels1, labels2)
nrep = 10
nboot = 100

segment_stabilities = []
for n_segments in range(2, 9):
    bootstrap_stabilities = []
    for _ in range(nboot):
      
        bootstrap_sample = resample(MD_x, random_state=1234) 
        labels_sample = perform_kmeans(bootstrap_sample, n_segments)
   labels_original = perform_kmeans(MD_x, n_segments) 
        stability = calculate_stability(labels_original, labels_sample) bootstrap_stabilities.append(stability)   segment_stabilities.append(bootstrap_stabilities)
plt.boxplot(segment_stabilities, labels=range(2, 9), vert=False)
plt.xlabel("Number of Segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Global Stability Analysis")
plt.show()