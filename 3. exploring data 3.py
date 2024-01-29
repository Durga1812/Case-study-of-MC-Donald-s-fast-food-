import pandas as pd
from sklearn.decomposition import PCA
segmentation matrix named MD_x
pca = PCA()
MD_pca_result = pca.fit_transform(MD_x) 
print("Importance of components:")
print(pd.DataFrame({
    'Standard deviation': pca.explained_variance_,
    'Proportion of Variance': pca.explained_variance_ratio_,
    'Cumulative Proportion': pca.explained_variance_ratio_.cumsum()
}, index=[f'PC{i+1}' for i in range(MD_x.shape[1])]))