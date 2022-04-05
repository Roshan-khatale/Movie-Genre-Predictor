import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/movie_plots.csv")
X = TfidfVectorizer().fit_transform(df["plot"])
y = df["genre"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump((TfidfVectorizer(), model), "models/movie_genre_model.pkl")
print("Model saved.")