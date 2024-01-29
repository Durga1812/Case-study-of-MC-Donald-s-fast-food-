import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
 segmentation matrix named MD_x
pca = PCA()
MD_pca_result = pca.fit_transform(MD_x)
projected_data = pca.transform(MD_x)
plt.scatter(projected_data[:, 0], projected_data[:, 1], color='grey') 
for i, feature in enumerate(segmentation_data.columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='red', head_width=0.05, head_length=0.05)
    plt.text(pca.components_[0, i], pca.components_[1, i], feature, color='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projected Data with Original Segmentation Variables')
plt.show()