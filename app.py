# ================================
# IMDb Sentiment & Emotion Analyzer - Final Premium UI
# ================================

import streamlit as st
import pandas as pd
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import gdown
import os, base64

# Setup Page
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

# --------------------------------
# BACKGROUND IMAGE
# --------------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background("samuel-regan-asante-wMkaMXTJjlQ-unsplash.jpg")

# --------------------------------
# GLASS CONTAINER STYLE
# --------------------------------
st.markdown("""
<style>
.glass-box {
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 45px;
    margin-top: 60px;
    margin-left: auto;
    margin-right: auto;
    max-width: 1100px;
    box-shadow: 0 8px 45px rgba(0,0,0,0.55);
    color: white !important;
}

/* Movie Recommendation Cards */
.movie-card {
    background: rgba(255,255,255,0.12);
    padding: 15px;
    border-radius: 14px;
    text-align: center;
    margin: 6px;
    font-size: 17px;
    border: 1px solid rgba(255,255,255,0.18);
    transition: 0.3s;
    cursor: pointer;
}
.movie-card:hover {
    background: rgba(255,255,255,0.25);
    transform: translateY(-4px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.title("🎬 IMDb Movie Review - NLP Sentiment & Emotion Analyzer")
st.write("Analyze any movie review — get its **sentiment**, **dominant emotion**, and **visual insights**!")

# --------------------------------
# LOAD DATASET
# --------------------------------
file_id = "1c-6qg1kGsuDrNXS9iCH4L1ryqiARMAUM"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("IMDB_Dataset.csv"):
    gdown.download(url, "IMDB_Dataset.csv", quiet=False)

data = pd.read_csv("IMDB_Dataset.csv", on_bad_lines='skip')
data = data.sample(5000, random_state=42)

# MODEL
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(data['review'])
model = LogisticRegression(max_iter=200).fit(X, data['sentiment'])

# --------------------------------
# USER INPUT
# --------------------------------
review = st.text_area("📝 Enter a movie review:", height=150)

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please type something first.")
    else:
        prediction = model.predict(tfidf.transform([review]))[0]
        emo = NRCLex(review)
        scores = emo.raw_emotion_scores

        st.subheader("🧠 Analysis Result")
        st.write(f"**Sentiment:** {prediction.capitalize()}")
        st.write(f"**Dominant Emotion:** {max(scores, key=scores.get).capitalize() if scores else 'Neutral'}")

        # Emotion Chart
        if scores:
            df = pd.DataFrame(scores.items(), columns=['Emotion', 'Score']).sort_values("Score", ascending=False)
            st.subheader("🎭 Emotion Breakdown")
            st.bar_chart(df.set_index("Emotion"))

        # MOVIE RECOMMENDATION
        st.subheader("🍿 Movie Recommendations")

        movies = {
            "positive": ["The Pursuit of Happyness","Forrest Gump","Coco","Interstellar","Inside Out","La La Land","3 Idiots","Zindagi Na Milegi Dobara"],
            "negative": ["Joker","Fight Club","Requiem for a Dream","The Whale","The Green Mile","Manchester by the Sea","Grave of the Fireflies","Black Swan"],
            "neutral": ["Inception","Tenet","Arrival","The Prestige","Her","Source Code","Eternal Sunshine of the Spotless Mind","The Matrix"]
        }

        rec_list = movies.get(prediction, movies["neutral"])

        cols = st.columns(4)
        for i, m in enumerate(rec_list):
            cols[i % 4].markdown(f"<div class='movie-card'>🍿 {m}</div>", unsafe_allow_html=True)

# --------------------------------
# WORDCLOUD SECTION
# --------------------------------
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")

pos = " ".join(data[data.sentiment=="positive"].review)
neg = " ".join(data[data.sentiment=="negative"].review)

fig, ax = plt.subplots(1,2,figsize=(13,5))
ax[0].imshow(WordCloud(colormap='Greens', background_color='white').generate(pos))
ax[0].set_title("Positive Reviews")
ax[0].axis("off")

ax[1].imshow(WordCloud(colormap='Reds', background_color='white').generate(neg))
ax[1].set_title("Negative Reviews")
ax[1].axis("off")

st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
