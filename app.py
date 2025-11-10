import streamlit as st
import pandas as pd
from nrclex import NRCLex
from nltk.tokenize import word_tokenize
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
        # Get sentiment probabilities
        proba = model.predict_proba(user_vector)[0]
        pos_score = proba[list(model.classes_).index("positive")]
        neg_score = proba[list(model.classes_).index("negative")]

        # Neutral threshold check
        if abs(pos_score - neg_score) < 0.15:
            sentiment = "neutral"
        else:
            sentiment = model.predict(user_vector)[0]
        
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
        tokens = word_tokenize(review)
        clean_text = " ".join(tokens)
        emo = NRCLex(review)
        scores = emo.raw_emotion_scores

        # --- Neutral Detection Logic ---
        total = sum(scores.values())
        if total == 0:
            dominant = "Neutral"
        else:
            # Normalize scores
            percentages = {emotion: (count / total) for emotion, count in scores.items()}

            # If no emotion strongly dominates -> Neutral
            if max(percentages.values()) < 0.35:
                dominant = "Neutral"
            else:
                dominant = max(percentages.items(), key=lambda x: x[1])[0]

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
        # ================================
        # 🎬 Curated Premium Recommendations (Fixed List + TMDB Posters)
        # ================================
        import requests

        TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

        # Curated lists (Best choices)
        curated_lists = {
            "positive": [
                "Interstellar", "Whiplash", "3 Idiots", "Forrest Gump",
                "The Pursuit of Happyness", "Coco", "Zindagi Na Milegi Dobara",
                "La La Land", "Soul", "Good Will Hunting",
                "The Good Place", "The Secret Life of Walter Mitty",
                "Inside Out", "The Shawshank Redemption", "Ratatouille", "Life of Pi"
            ],
            "negative": [
                "Joker", "Requiem for a Dream", "The Whale", "Manchester by the Sea",
                "Grave of the Fireflies", "A Silent Voice", "Black Swan",
                "BoJack Horseman", "Eternal Sunshine of the Spotless Mind",
                "Taxi Driver", "Blue Valentine", "Fight Club",
                "Melancholia", "The Green Mile", "The Pianist", "Schindler's List"
            ],
            "neutral": [
                "Inception", "The Matrix", "Tenet", "Arrival",
                "The Prestige", "Her", "Blade Runner 2049",
                "Dark", "Black Mirror", "Mr. Robot",
                "Source Code", "Gone Girl", "Interstellar", 
                "Shutter Island", "No Country for Old Men", "Dune"
            ]
        }

        def fetch_tmdb_data(title):
            """Search TMDB and return proper poster + metadata"""
            search_url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={title}"
            response = requests.get(search_url).json()
            results = response.get("results", [])
    
            if not results:
                return None
    
            item = results[0]  # Best match
            poster = f"https://image.tmdb.org/t/p/w500{item['poster_path']}" if item.get("poster_path") else None
            rating = item.get("vote_average", "N/A")
    
            # Get year
            date = item.get("release_date") or item.get("first_air_date") or ""
            year = date.split("-")[0] if date else "N/A"
    
            # Get genres (extra polish)
            genre_url = f"https://api.themoviedb.org/3/{item['media_type']}/{item['id']}?api_key={TMDB_API_KEY}&language=en-US"
            genre_res = requests.get(genre_url).json()
            genres = ", ".join([g["name"] for g in genre_res.get("genres", [])])
    
            return {
                "title": f"{title} ({year})",
                "poster": poster,
                "rating": round(rating, 1),
                "genres": genres or "N/A"
            }

        # Display Section
        st.subheader("🍿 Recommended Movies & TV Shows For Your Mood")

        selected_list = curated_lists.get(sentiment.lower(), curated_lists["neutral"])
        items = [fetch_tmdb_data(title) for title in selected_list]
        items = [i for i in items if i]  # remove None results

        cols = st.columns(4)
        for i, m in enumerate(items[:16]):  # show only 16
            with cols[i % 4]:
                if m["poster"]:
                    st.image(m["poster"], use_column_width=True)
                st.markdown(f"**{m['title']}**")
                st.write(f"⭐ {m['rating']}")
                st.write(f"🎭 {m['genres']}")

# ---------- WordCloud ----------
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")
pos = " ".join(data[data.sentiment == "positive"].review)
neg = " ".join(data[data.sentiment == "negative"].review)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(WordCloud(background_color="white", colormap="Greens").generate(pos)); ax[0].axis("off"); ax[0].set_title("Positive Reviews")
ax[1].imshow(WordCloud(background_color="white", colormap="Reds").generate(neg)); ax[1].axis("off"); ax[1].set_title("Negative Reviews")
st.pyplot(fig)
