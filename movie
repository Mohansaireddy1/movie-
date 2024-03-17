import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset of films related to sustainable urban mobility
films_df = pd.read_csv('films_dataset.csv')

# Preprocess the text data (film descriptions, genres, etc.)
films_df['combined_features'] = films_df['description'] + " " + films_df['genres']
films_df['combined_features'] = films_df['combined_features'].fillna('')  # Handle missing values

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(films_df['combined_features'])

# Function to get movie recommendations based on user preferences
def get_recommendations(user_preferences):
    # Transform user preferences into TF-IDF vectors
    user_preferences_tfidf = tfidf_vectorizer.transform([user_preferences])

    # Calculate similarity scores (cosine similarity) between user preferences and films
    cosine_similarities = linear_kernel(user_preferences_tfidf, tfidf_matrix).flatten()

    # Get indices of films sorted by similarity scores (descending order)
    similar_indices = cosine_similarities.argsort()[::-1]

    # Recommend top N films (excluding the input film itself)
    top_N = 5
    recommended_films = []
    for i in range(1, top_N + 1):
        film_index = similar_indices[i]
        recommended_films.append(films_df.iloc[film_index]['title'])

    return recommended_films

# Example usage:
user_preferences = "Documentary about sustainable transportation"
recommendations = get_recommendations(user_preferences)
print("Recommended Films:")
for film in recommendations:
    print("-", film)
