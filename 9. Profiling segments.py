import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

mosaic_df = pd.crosstab(mcdonalds['Segment'], pd.cut(mcdonalds['Like.n'], bins=[-float('inf'), 0, float('inf')], labels=['Hate', 'Love']))
plt.figure(figsize=(8, 6))
sns.heatmap(mosaic_df, annot=True, cmap='coolwarm', fmt='d', cbar=False)
plt.xlabel('Loving or Hating McDonald’s')
plt.ylabel('Segment Number')
plt.title('Mosaic Plot - Association between Segment Membership and Loving or Hating McDonald’s')
plt.show()

gender_mosaic_df = pd.crosstab(mcdonalds['Segment'], mcdonalds['Gender'])
plt.figure(figsize=(8, 6))
sns.heatmap(gender_mosaic_df, annot=True, cmap='coolwarm', fmt='d', cbar=False)
plt.title('Mosaic Plot - Gender Distribution across Segments')
plt.show()

 Parallel Box-and-Whisker Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Segment', y='Age', data=mcdonalds, notch=True, var_width=True)
plt.title('Association of Age with Segment Membership')
plt.show()
X = mcdonalds[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = (mcdonalds['Segment'] == 3).astype(int)  # Binary classification for Segment 3 membership

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

fig = px.plot_tree(tree, feature_names=X.columns, class_names=['Not Segment 3', 'Segment 3'])
fig.show()