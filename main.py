import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

try:
    df = pd.read_csv("condensed_data.csv", lineterminator="\n")
    with open("assets/sparse_overview.pickle", 'rb') as f:
        sparse_overview = pickle.load(f)
    with open('assets/vectorizer.pickle', 'rb') as v:
        vectorizer = pickle.load(v)
except FileNotFoundError:
    print("Generating required files...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
    vectorizer = vectorizer.fit(df["combined"])
    sparse_overview = vectorizer.fit_transform(df["combined"])
    

def get_movie_recommendations(user_movie, top_k=10):
    cbf_df = df.drop(["combined"], axis=1)
    user_profile = df[df["title"] == user_movie]["combined"]
    # Convert user profile to feature vector
    user_features = vectorizer.transform([user_profile.values[0]])

    # Calculate similarity between user profile and all movies
    similarities = cosine_similarity(user_features, sparse_overview).flatten()

    # Sort movies based on similarity and get top recommendations
    indices = similarities.argsort()[::-1][:top_k]
    recommended_movies = cbf_df.iloc[indices]

    return recommended_movies


st.title("Movie Recommender System")
options = df['title'].tolist()
options.append('Type name here ...')

user_input = st.selectbox(
    'Enter the name of an movie you like:', options=options, index=len(options)-1
)

if user_input == options[-1]:
    pass
elif user_input:
    try:
        recommendations = get_movie_recommendations(user_input)
        st.write(f"Recommended movie similar to {user_input}:")
        st.table(recommendations)
    except KeyError:
        st.write(f"Sorry, {user_input} is not in our database.")
