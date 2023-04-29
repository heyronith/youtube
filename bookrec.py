import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack

# Load the books DataFrame
books_df = pd.read_csv(r"1.csv")

def preprocess_text(text):
    return str(text).lower().replace('[^a-z\s]', '')

books_df['processed_description'] = books_df['description'].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])

n_neighbors = 6  # Number of neighbors to return, including the input book
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

# Create a mapping from book titles to their index 
title_to_index = pd.Series(books_df.index, index=books_df['title'].apply(lambda x: x.lower()))

# Recommendation function
def recommend_books(title, model=model, books_df=books_df, title_to_index=title_to_index):
    idx = title_to_index[title.lower()]
    distances, indices = model.kneighbors(tfidf_matrix[idx])
    indices = indices[0][1:]  # Remove the input book from the recommendations
    return books_df['title'].iloc[indices].values.tolist()

def main():
    while True:
        book_title = input("Enter the book title (Type 'end to end the program):")
        if book_title.lower() == 'end':
            break
        if book_title.lower() not in title_to_index:
            print("Sorry ! book not found in the database ! Try another book !")
            continue

        recommendations = recommend_books(book_title)
        print("\n The following are the best recommendations for '{}':\n".format(book_title))
        print(recommendations)
        print("\n")

if __name__ == "__main__":
    main()
