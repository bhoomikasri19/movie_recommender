# 🎬 Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on genres, keywords, and plot overview.

---

## 🎯 How It Works

1. User selects a movie from the dropdown
2. The system finds movies with similar genres, keywords and plot
3. Returns top 5 most similar movies instantly

---

## 🧠 How It's Built

- Combined movie genres, keywords and overview into a single "tags" field
- Converted text to numerical vectors using **CountVectorizer**
- Calculated similarity between all movies using **Cosine Similarity**
- No model training needed — pure content-based filtering

---

## 💻 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (CountVectorizer, Cosine Similarity)
- Streamlit, Joblib

---

## 📊 Dataset
- **Source:** TMDB 5000 Movie Dataset (Kaggle)
- **Size:** 2000 most popular movies
- **Features used:** Title, Overview, Genres, Keywords

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/bhoomikasri19/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r requirements.txt

# Train the model first
python test.py

# Run the app
streamlit run app.py
```

> **Note:** Dataset not included due to size.
> Download from Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
> Place `tmdb_5000_movies.csv` in root folder and run `python test.py`

---

## 📁 Project Structure

```
movie-recommender/
├── app.py              # Streamlit web app
├── test.py             # Data processing & similarity matrix
├── requirements.txt    # Dependencies
├── README.md
└── model/
    ├── movies.pkl      # Processed movies dataframe
    └── similarity.pkl  # Cosine similarity matrix
```

---

## 🌐 Live Demo
https://movierecommender-ml.streamlit.app/

---

