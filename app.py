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

        st.subheader("🍿 Movie Recommendations for You")
        movie_list = {
            "positive": ["Interstellar", "Coco", "3 Idiots", "La La Land", "Zindagi Na Milegi Dobara"],
            "negative": ["Joker", "Fight Club", "The Whale", "Requiem for a Dream"],
            "neutral": ["Inception", "Arrival", "The Matrix", "Her", "Tenet"]
        }
        for m in movie_list.get(sentiment, movie_list["neutral"]):
            st.write("🎞️", m)


# ---------- WordCloud ----------
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")
pos = " ".join(data[data.sentiment == "positive"].review)
neg = " ".join(data[data.sentiment == "negative"].review)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(WordCloud(background_color="white", colormap="Greens").generate(pos)); ax[0].axis("off"); ax[0].set_title("Positive Reviews")
ax[1].imshow(WordCloud(background_color="white", colormap="Reds").generate(neg)); ax[1].axis("off"); ax[1].set_title("Negative Reviews")
st.pyplot(fig)
