from statistics import quantiles

import pandas as pd
from matplotlib import pyplot as plt
from numpy import quantile
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

file_path = "music_dataset.csv"
df = pd.read_csv(file_path)
# print("First 5 records:\n", df.head())

# Top 10 genres bar chart
top_genres = df['Genre'].value_counts().head(10)
plt.figure(figsize = (10,5))
top_genres.plot(kind = 'bar')

plt.xlabel("Genre")
plt.ylabel("Number of Songs")
plt.title("Top 10 Genres")
plt.xticks(rotation = 360)
plt.tight_layout()
plt.show()

# Line chart for average daily streams by release year
avg_streams = df.groupby('Release Year')['Daily Streams'].mean()
plt.figure(figsize = (10,5))
avg_streams.plot(kind = 'line', marker = 'o')

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

# Define threshold (top 25% of streams)
threshold = df['Streams'].quantile(0.75)

# Define features
x = df[['Danceability', 'Energy', 'Acousticness', 'TikTok Virality', 'Lyrics Sentiment']]
y = df['Streams'].apply(lambda x: 1 if x >= threshold else 0)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Initialize and train model
model = RandomForestClassifier(n_estimators = 100, random_state = 0)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))