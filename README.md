# Content-Based Movie Recommendation System

A content-based movie recommendation engine built using Python, Pandas, and Scikit-learn. This system analyzes movie metadata—including genres, plot overviews, keywords, top cast, and directors—from the TMDB 5000 dataset to suggest contextually similar movies.

---

## Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning / NLP:** `scikit-learn` (`CountVectorizer`, `cosine_similarity`)
* **Utilities:** `ast` (for parsing JSON-like string data)

---

## Dataset
The project utilizes two datasets from the TMDB 5000 Movie Dataset:
1. `tmdb_5000_movies.csv`: Contains movie details like budget, genres, ID, keywords, overview, popularity, release date, revenue, runtime, and title.
2. `tmdb_5000_credits.csv`: Contains cast and crew information mapped to each movie.

---

## Project Workflow
1. **Data Merging:** Combined the `movies` and `credits` datasets on the movie title and filtered out unnecessary columns.
2. **Metadata Parsing:** Extracted valuable features from stringified JSON columns (pulling individual genre names, top 3 cast members, and the director).
3. **Text Preprocessing:** Lowercased text and removed spaces between multi-word entities (e.g., transforming `"Sam Worthington"` to `"SamWorthington"`) so the vectorizer treats them as unique tokens.
4. **Tag Generation:** Merged the overview, genres, keywords, cast, and crew into a single comprehensive `tags` text column for each movie.
5. **Vectorization:** Applied **Bag-of-Words (BoW)** via Scikit-learn's `CountVectorizer` (limiting to the top 5,000 features while removing English stop words) to convert text tags into numerical vectors.
6. **Similarity Calculation:** Computed **Cosine Similarity** across the vector matrix to score how closely related movies are to one another.

---

## Output

Recommendations for 'Avatar':
- Titan A.E.
- Small Soldiers
- Independence Day
- E.T. the Extra-Terrestrial
- Aliens

## Getting Started

### 1. Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn

import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2. Merge and clean data
movies = movies.merge(credits, on='title')
movies = movies[
    ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
]
movies.dropna(inplace=True)


# 3. Helper functions for parsing JSON strings
def convert(obj):
  return [i['name'] for i in ast.literal_eval(obj)]


def convert_cast(obj):
  return [i['name'] for i in ast.literal_eval(obj)][:3]


def fetch_director(obj):
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      return [i['name']]
  return []


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# 4. Remove spaces between multi-word names
def collapse(L):
  return [i.replace(' ', '') for i in L]


movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# 5. Create a combined tags column
movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df['tags'] = new_df['tags'].str.lower()

# 6. Vectorization & Cosine Similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)


# 7. Recommendation function
def recommend(movie):
  try:
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    print(f"Recommendations for '{movie}':")
    for i in movies_list:
      print(f'- {new_df.iloc[i[0]].title}')
  except IndexError:
    print(f"Movie '{movie}' not found in database.")


# Example Usage:
recommend('Avatar')
