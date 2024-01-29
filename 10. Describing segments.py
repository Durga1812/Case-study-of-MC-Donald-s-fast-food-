import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

visit_mean = mcdonalds.groupby('Segment')['VisitFrequency'].mean()
like_mean = mcdonalds.groupby('Segment')['Like.n'].mean()
female_percentage = mcdonalds.groupby('Segment')['Gender'].apply(lambda x: (x == 'Female').mean()) 
plt.figure(figsize=(10, 8))
sns.scatterplot(x=visit_mean, y=like_mean, size=female_percentage * 10, sizes=(50, 200), hue=visit_mean.index)
 enumerate(visit_mean.index):
    plt.annotate(txt, (visit_mean[i], like_mean[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.xlim(2, 4.5)
plt.ylim(-3, 3)

plt.xlabel('Visit Frequency')
plt.ylabel('Liking McDonald’s')
plt.title('Segment Evaluation Plot')
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()