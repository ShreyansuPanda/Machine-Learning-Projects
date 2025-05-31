import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the datasets
movies = pd.read_csv("movies.csv")  
ratings = pd.read_csv("ratings.csv")  

# Create User-Item Matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


sparse_matrix = csr_matrix(user_movie_matrix.values)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(sparse_matrix)

# Function to recommend movies
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix.index:
        return "User ID not found!"

    user_index = user_movie_matrix.index.get_loc(user_id)
    
    # Find similar users (FIXED DIMENSION ERROR)
    distances, indices = knn.kneighbors(sparse_matrix[user_index], n_neighbors=6)
    
    similar_users = indices.flatten()[1:]  

    # Get movies rated highly by similar users
    recommended_movies = []
    for sim_user in similar_users:
        top_movies = user_movie_matrix.iloc[sim_user].sort_values(ascending=False).index[:num_recommendations]
        recommended_movies.extend(top_movies)

    recommended_movies = list(set(recommended_movies))[:num_recommendations]

    # Map movie IDs to titles
    movie_titles = movies[movies['movieId'].isin(recommended_movies)]['title'].tolist()
    return movie_titles

# Test the recommendation system
user_id = 3
print(f"Recommended movies for User {user_id}: {recommend_movies(user_id)}")
