import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from stability_selection import StabilitySelection

np.random.seed(1234)
MD_x = pd.DataFrame(np.random.rand(100, 10), columns=['feature_{}'.format(i) for i in range(10)])
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    labels = kmeans.fit_predict(data)
    return labels
nrep = 10

segmentations = {}
for n_segments in range(2, 9):
    labels_list = [perform_kmeans(MD_x, n_segments) for _ in range(nrep)]
    segmentations[n_segments] = labels_list
stability_selection = StabilitySelection()
stability_scores = stability_selection.fit(np.array(list(segmentations.values())))

optimal_segments = np.argmax(stability_scores) + 2  # +2 because the loop starts from 2
print("Optimal number of segments:", optimal_segments)