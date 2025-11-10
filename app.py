import streamlit as st
import pandas as pd
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import base64
import gdown
import os

# ---------- Background Function ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{encoded}) no-repeat center center fixed;
            background-size: cover;
        }}

        /* Glass effect container */
        .block-container {{
            background: rgba(0, 0, 0, 0.55);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            padding: 45px 50px;
            border-radius: 20px;
            box-shadow: 0 0 40px rgba(0,0,0,0.5);
            max-width: 950px;
        }}

        h1, h2, h3, p, label {{
            color: white !important;
            text-shadow: 0px 0px 6px rgba(0,0,0,1);
        }}

        .stTextInput textarea {{
            background: rgba(255,255,255,0.2) !important;
            color: white !important;
        }}

        .stButton button {{
            background-color: #ff4757;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }}

        .stButton button:hover {{
            background-color: #ff6b81;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------- SET BACKGROUND ----------
set_bg("samuel-regan-asante-wMkaMXTJjlQ-unsplash.jpg")

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="Sentiment & Emotion Analyzer")

# ---------- NLTK Downloads ----------
nltk.download('punkt', quiet=True)

# ---------- Title ----------
st.title("🎬 IMDb Movie Review - NLP Sentiment & Emotion Analyzer")
st.write("Analyze any movie review — get its **sentiment**, **dominant emotion**, and personalized **movie recommendations**!")

# ---------- Load Dataset from Google Drive ----------
file_id = "1c-6qg1kGsuDrNXS9iCH4L1ryqiARMAUM"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("IMDB_Dataset.csv"):
    gdown.download(url, "IMDB_Dataset.csv", quiet=False)

data = pd.read_csv("IMDB_Dataset.csv", on_bad_lines="skip").sample(5000, random_state=42)

# ---------- Train Model ----------
X = data["review"]
y = data["sentiment"]
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)
model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# ---------- User Input ----------
review = st.text_area("✍️ Enter a movie review:", height=150)

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("⚠️ Please type a review first!")
    else:
        user_vector = tfidf.transform([review])
        sentiment = model.predict(user_vector)[0]

        emo = NRCLex(review)
        scores = emo.raw_emotion_scores
        dominant = max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"

        st.subheader("🧠 Analysis Result")
        st.write(f"**Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Dominant Emotion:** {dominant.capitalize()}")

        if sentiment == "positive":
            st.balloons()
        elif sentiment == "negative":
            st.snow()

        if scores:
            st.subheader("🎭 Emotion Breakdown")
            emo_df = pd.DataFrame(scores.items(), columns=["Emotion", "Score"]).sort_values("Score", ascending=False)
            st.bar_chart(emo_df.set_index("Emotion"))

                # ================================
                # ================================
        # 🎬 Movie Recommendations (TMDB API + Posters)
        # ================================
        import requests

       st.subheader("🍿 Movie & TV Show Recommendations For You")

# NEW curated recommendation list
recommendations = {
    "positive": [
        {
            "title": "Interstellar (2014)",
            "poster": "https://m.media-amazon.com/images/I/71niXI3lxlL._AC_SY550_.jpg",
            "rating": "8.7",
            "genres": "Adventure, Drama, Sci-Fi"
        },
        {
            "title": "Whiplash (2014)",
            "poster": "https://m.media-amazon.com/images/I/81p+xe8cbnL._AC_SY550_.jpg",
            "rating": "8.5",
            "genres": "Drama, Music"
        },
        {
            "title": "3 Idiots (2009)",
            "poster": "https://m.media-amazon.com/images/I/71i+6k8EGFL._AC_SY679_.jpg",
            "rating": "8.4",
            "genres": "Comedy, Drama"
        },
        {
            "title": "Forrest Gump (1994)",
            "poster": "https://m.media-amazon.com/images/I/61xFddgET-L._AC_SY741_.jpg",
            "rating": "8.8",
            "genres": "Drama, Romance"
        },
        {
            "title": "The Pursuit of Happyness (2006)",
            "poster": "https://m.media-amazon.com/images/I/71U42DXSxPL._AC_SY679_.jpg",
            "rating": "8.0",
            "genres": "Biography, Drama"
        },
        {
            "title": "Zindagi Na Milegi Dobara (2011)",
            "poster": "https://m.media-amazon.com/images/I/71Q1Iu4su9L._AC_SY679_.jpg",
            "rating": "8.2",
            "genres": "Adventure, Comedy, Drama"
        },
        {
            "title": "Coco (2017)",
            "poster": "https://m.media-amazon.com/images/I/81cgG2YZ3JL._AC_SY679_.jpg",
            "rating": "8.4",
            "genres": "Animation, Adventure, Family"
        },
        {
            "title": "The Good Place (Series, 2016–2020)",
            "poster": "https://m.media-amazon.com/images/I/71pQYp9GhFL._AC_SY679_.jpg",
            "rating": "8.2",
            "genres": "Comedy, Fantasy"
        }
    ],

    "negative": [
        {
            "title": "Joker (2019)",
            "poster": "https://m.media-amazon.com/images/I/71xYLsRzJ-L._AC_SY679_.jpg",
            "rating": "8.4",
            "genres": "Crime, Drama, Thriller"
        },
        {
            "title": "Requiem for a Dream (2000)",
            "poster": "https://m.media-amazon.com/images/I/71UgHTA+3zL._AC_SY679_.jpg",
            "rating": "8.3",
            "genres": "Drama"
        },
        {
            "title": "Manchester by the Sea (2016)",
            "poster": "https://m.media-amazon.com/images/I/71AqOD7LYXL._AC_SY679_.jpg",
            "rating": "7.8",
            "genres": "Drama"
        },
        {
            "title": "The Whale (2022)",
            "poster": "https://m.media-amazon.com/images/I/81rE5c8uXwL._AC_SY679_.jpg",
            "rating": "7.7",
            "genres": "Drama"
        },
        {
            "title": "BoJack Horseman (Series, 2014–2020)",
            "poster": "https://m.media-amazon.com/images/I/81Z0g3UkH2L._AC_SY679_.jpg",
            "rating": "8.8",
            "genres": "Animation, Comedy, Drama"
        },
        {
            "title": "Eternal Sunshine of the Spotless Mind (2004)",
            "poster": "https://m.media-amazon.com/images/I/71U9SN7VrYL._AC_SY679_.jpg",
            "rating": "8.3",
            "genres": "Drama, Romance, Sci-Fi"
        }
    ],

    "neutral": [
        {
            "title": "Inception (2010)",
            "poster": "https://m.media-amazon.com/images/I/71Hmy8vTOJL._AC_SY679_.jpg",
            "rating": "8.8",
            "genres": "Action, Sci-Fi, Thriller"
        },
        {
            "title": "The Matrix (1999)",
            "poster": "https://m.media-amazon.com/images/I/51EG732BV3L.jpg",
            "rating": "8.7",
            "genres": "Action, Sci-Fi"
        },
        {
            "title": "Her (2013)",
            "poster": "https://m.media-amazon.com/images/I/71o9i5-2C5L._AC_SY679_.jpg",
            "rating": "8.0",
            "genres": "Drama, Romance, Sci-Fi"
        },
        {
            "title": "Black Mirror (Series, 2011– )",
            "poster": "https://m.media-amazon.com/images/I/81TZ5MMEZeL._AC_SY679_.jpg",
            "rating": "8.8",
            "genres": "Drama, Sci-Fi, Thriller"
        },
        {
            "title": "Tenet (2020)",
            "poster": "https://m.media-amazon.com/images/I/71tV4GN28uL._AC_SY679_.jpg",
            "rating": "7.3",
            "genres": "Action, Sci-Fi, Thriller"
        },
        {
            "title": "The Prestige (2006)",
            "poster": "https://m.media-amazon.com/images/I/71yRk7P3tJL._AC_SY679_.jpg",
            "rating": "8.5",
            "genres": "Drama, Mystery, Thriller"
        }
    ]
}

# Display recommendations
items = recommendations.get(sentiment.lower(), recommendations["neutral"])
cols = st.columns(4)
for i, m in enumerate(items):
    with cols[i % 4]:
        st.image(m["poster"], use_column_width=True)
        st.markdown(f"**{m['title']}**")
        st.write(f"⭐ Rating: {m['rating']}")
        st.write(f"🎭 Genres: {m['genres']}")

# ---------- WordCloud ----------
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")
pos = " ".join(data[data.sentiment == "positive"].review)
neg = " ".join(data[data.sentiment == "negative"].review)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(WordCloud(background_color="white", colormap="Greens").generate(pos)); ax[0].axis("off"); ax[0].set_title("Positive Reviews")
ax[1].imshow(WordCloud(background_color="white", colormap="Reds").generate(neg)); ax[1].axis("off"); ax[1].set_title("Negative Reviews")
st.pyplot(fig)
