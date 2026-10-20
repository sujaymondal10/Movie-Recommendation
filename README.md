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
