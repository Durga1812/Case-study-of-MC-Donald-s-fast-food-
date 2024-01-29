import numpy as np
import pandas as pd
from mixtools import MixModel
from sklearn.cluster import KMeans

k_range = range(2, 9)
n_restarts = 10

aic_values = []
bic_values = []
icl_values = []

for k in k_range:
    mix_model = MixModel(data['MD.x'].values.reshape(-1, 1), n_components=k, n_init=n_restarts)
    aic_values.append(mix_model.aic())
    bic_values.append(mix_model.bic())
    icl_values.append(mix_model.icl())

print("AIC values:", aic_values)
print("BIC values:", bic_values)
print("ICL values:", icl_values) 
import matplotlib.pyplot as plt

plt.plot(k_range, aic_values, label='AIC')
plt.plot(k_range, bic_values, label='BIC')
plt.plot(k_range, icl_values, label='ICL')
plt.xlabel('Number of Segments')
plt.ylabel('Value of Information Criteria')
plt.legend()
plt.show()

# Choose the number of segments based on the plot (e.g., 4 segments)
chosen_segments = 4

final_mix_model = MixModel(data['MD.x'].values.reshape(-1, 1), n_components=chosen_segments, n_init=n_restarts)
kmeans = KMeans(n_clusters=chosen_segments, random_state=1234)
data['kmeans_clusters'] = kmeans.fit_predict(data['MD.x'].values.reshape(-1, 1))

cross_tab = pd.crosstab(data['kmeans_clusters'], final_mix_model.predict(), margins=True)
print(cross_tab)