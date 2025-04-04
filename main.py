import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

file_path = "music_dataset.csv"
df = pd.read_csv(file_path)
# print("First 5 records:\n", df.head())

# Top 10 genres bar chart
top_genres = df['Genre'].value_counts().head(10)
plt.figure(figsize=(10,5))
top_genres.plot(kind='bar')

plt.xlabel("Genre")
plt.ylabel("Number of Songs")
plt.title("Top 10 Genres")
plt.xticks(rotation=360)
plt.tight_layout()
plt.show()

# Line chart for average daily streams by release year
avg_streams = df.groupby('Release Year')['Daily Streams'].mean()
plt.figure(figsize=(10,5))
avg_streams.plot(kind='line', marker='o')

plt.xlabel("Release Year")
plt.ylabel("Average Daily Streams")
plt.title("Average Daily Streams By Release Year")
plt.tight_layout()
plt.show()

# Horizontal bar graph showing top 10 streamed artists
df.groupby('Artist')['Streams'].sum().nlargest(10).plot(kind="barh", figsize=(10,5))
plt.xlabel("Streams")
plt.ylabel("Artist")
plt.title("Top 10 Artists by Streams")
plt.tight_layout()
# Put highest streamed artist on top
plt.gca().invert_yaxis()
plt.show()