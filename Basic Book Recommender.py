import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack

# Load the books DataFrame
books_df = pd.read_csv(r"C:\Users\15132\Desktop\Projects\Book Recommender\books_1.Best_Books_Ever.csv")

# Preprocess the text data (you may need to customize this step based on your data)
def preprocess_text(text):
    return str(text).lower().replace('[^a-z\s]', '')

books_df['processed_description'] = books_df['description'].apply(preprocess_text)

# Combine genres and descriptions
books_df['combined_features'] = books_df['genres'] + ' ' + books_df['processed_description']

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

# Fit NearestNeighbors model
n_neighbors = 6  # Number of neighbors to return, including the input book
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

# Create a mapping from book titles to their index in the DataFrame
title_to_index = pd.Series(books_df.index, index=books_df['title'].apply(lambda x: x.lower()))

# Recommendation function
def recommend_books(title, model=model, books_df=books_df, title_to_index=title_to_index):
    idx = title_to_index[title.lower()]
    distances, indices = model.kneighbors(tfidf_matrix[idx])
    indices = indices[0][1:]  # Remove the input book from the recommendations
    return books_df['title'].iloc[indices].values.tolist()

def main():
    while True:
        book_title = input("Enter a book title to get recommendations (type 'quit' to exit): ")
        if book_title.lower() == 'quit':
            break
        if book_title.lower() not in title_to_index:
            print("Book not found in the database. Please try another title.")
            continue

        recommendations = recommend_books(book_title)
        print("\nTop recommendations for '{}':\n".format(book_title))
        print(recommendations)
        print("\n")

if __name__ == "__main__":
    main()
