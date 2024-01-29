import pandas as pd
import numpy as np

pd.read_csv() or pd.read_excel() 
segmentation_data = mcdonalds.iloc[:, :11]
segmentation_matrix = segmentation_data.values
yes_matrix = segmentation_matrix == 'Yes'

# Convert TRUE to 1 and FALSE to 0
numeric_matrix = yes_matrix.astype(int)
average_values = np.mean(numeric_matrix, axis=0)
print("Average values for transformed segmentation variables:")
print(average_values)