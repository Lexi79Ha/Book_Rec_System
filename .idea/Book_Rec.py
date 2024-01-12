import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack
from langdetect import detect

books_df = pd.read_csv(r"book_rec.csv")

def preprocess_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

books_df['processed_description'] = books_df['description'].apply(preprocess_text)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])

n_neighbors = 6
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

title_to_index = pd.Series(books_df.index, index=books_df['title'].apply(lambda x: x.lower()))

from langdetect import DetectorFactory
DetectorFactory.seed = 0

genre_to_index = pd.Series(books_df.index, index=books_df['genres'].apply(lambda x: x.lower()))

def recommend_books(query, query_type='title', genre=None, model=model, books_df=books_df, title_to_index=title_to_index, genre_to_index=genre_to_index):
    if query_type == 'title':
        match = process.extractOne(query.lower(), title_to_index.index)
        idx = title_to_index[match[0]]
    elif query_type == 'genres':
        match = process.extractOne(query.lower(), genre_to_index.index)
        idx = genre_to_index[match[0]]
    else:
        print("Invalid query type. Please choose from 'title', 'author', or 'genres'.")
        return

    distances, indices = model.kneighbors(tfidf_matrix[idx])
    indices = indices[0][1:]
    recommendations = books_df['title'].iloc[indices].values.tolist()

    if genre:
        recommendations = [title for title in recommendations if genre.lower() in books_df.loc[books_df['title'] == title, 'genres'].values[0].lower()]

    # Filter out non-English books
    english_recommendations = [book for book in recommendations if detect(book) == 'en']

    filtered_recommendations = [book for book in english_recommendations if
                                detect(book) == 'en' and not book.endswith('eSampler')]

    return filtered_recommendations


def top_rated_in_genre(genres, n=5):
    genres = [genre.strip() for genre in genres.split(',')]
    genre_books = books_df[books_df['genres'].apply(lambda x: all(genre.lower() in x.lower() for genre in genres))]
    top_rated = genre_books.nlargest(n, 'rating')
    return top_rated['title'].values.tolist()

search_type = input("Enter the type of search you want to perform (title, or genre): ")

while search_type.lower() not in ['title', 'genre']:
    print("Invalid search type. Please choose from 'title', or 'genre'.")
    search_type = input("Enter the type of search you want to perform (title, or genre): ")

query = input("Enter your {}: ".format(search_type))

while True:
    if search_type.lower() == 'title':
        if query.lower() in title_to_index:
            recommendations = recommend_books(query, query_type='title')
            if recommendations:
                print("\nThe following are the best recommendations for '{}':\n".format(query))
                for i, book in enumerate(recommendations, 1):
                    print("{}. {}".format(i, book))
                print("\n")
                break
            else:
                print("Sorry! That book is not found in our database. Please check for spelling errors or try another book!")
        else:
            print("Sorry! That book is not found in our database. Please check for spelling errors or try another book!")
    elif search_type.lower() == 'genre':
        top_rated_books = top_rated_in_genre(query)
        if top_rated_books:
            print("\nThe following are the top-rated books in the '{}' genre:\n".format(query))
            for i, book in enumerate(top_rated_books, 1):
                print("{}. {}".format(i, book))
            print("\n")
            break
        else:
            print("Sorry! We could not find any book recommendations under that genre. If searching multiple genres at once please seperate by comma or try another genre!")
    else:
        print("Invalid search type. Please choose from 'title', or 'genre'.")

    query = input("Enter a new {}: ".format(search_type))