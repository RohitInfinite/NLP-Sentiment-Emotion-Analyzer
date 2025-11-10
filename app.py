import streamlit as st
import pandas as pd
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import gdown
import os
import base64

# -----------------------------------------
# PAGE SETTINGS (ye sabse upar hona hi chahiye)
# -----------------------------------------
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="wide")

# -----------------------------------------
# BACKGROUND IMAGE
# -----------------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg("samuel-regan-asante-wMkaMXTJjlQ-unsplash.jpg")  # <<======= CHANGE THIS TO YOUR IMAGE NAME

# -----------------------------------------
# GLOBAL GLASS STYLES + CARD SYSTEM
# -----------------------------------------
st.markdown("""
<style>
.hero {
    background: rgba(0,0,0,0.35);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 40px;
    margin-top: 40px;
    text-align: center;
}
.card {
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 30px;
    margin-top: 30px;
}
h1, h2, h3, p, label {
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.9);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HERO SECTION (top intro block)
# -----------------------------------------
st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.title("🎬 IMDb Movie Review - NLP Sentiment & Emotion Analyzer")
st.write("Analyze any movie review to reveal its **mood**, **emotions**, and get fun **movie recommendations** 🎭")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# DOWNLOAD + PREPARE DATA
# -----------------------------------------
file_id = "1c-6qg1kGsuDrNXS9iCH4L1ryqiARMAUM"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("IMDB_Dataset.csv"):
    gdown.download(url, "IMDB_Dataset.csv", quiet=False)

data = pd.read_csv("IMDB_Dataset.csv", on_bad_lines='skip')
data = data.sample(5000, random_state=42)

X = data['review']
y = data['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# -----------------------------------------
# REVIEW INPUT CARD
# -----------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
user_input = st.text_area("✍️ Enter a movie review:", height=150)
if st.button("Analyze Review"):
    if user_input.strip():
        user_tfidf = tfidf.transform([user_input])
        sentiment = model.predict(user_tfidf)[0]

        emo = NRCLex(user_input)
        scores = emo.raw_emotion_scores
        dominant_emotion = max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"

        st.subheader("🧠 Analysis Result")
        st.write(f"**Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Dominant Emotion:** {dominant_emotion.capitalize()}")

        if sentiment == "positive":
            st.balloons()
        elif sentiment == "negative":
            st.snow()

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# EMOTION GRAPH CARD
# -----------------------------------------
if user_input.strip():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎭 Emotion Breakdown")
    emo_df = pd.DataFrame(scores.items(), columns=['Emotion', 'Score']).sort_values(by='Score', ascending=False)
    st.bar_chart(emo_df.set_index('Emotion'))
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# MOVIE RECOMMENDATION CARD
# -----------------------------------------
if user_input.strip():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🍿 Movie Recommendations (Based on Mood)")

    recs = {
        "positive": ["The Pursuit of Happyness", "Forrest Gump", "Interstellar", "Coco", "La La Land", "3 Idiots", "ZNMD"],
        "negative": ["Joker", "The Green Mile", "Fight Club", "The Whale", "Requiem for a Dream", "Manchester by the Sea"],
        "neutral": ["Inception", "The Prestige", "Arrival", "Her", "The Matrix", "Source Code"]
    }

    for movie in recs.get(sentiment, recs["neutral"]):
        st.write(f"🎞️ {movie}")

    st.success("Hope these match your vibe! 🍿✨")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# WORDCLOUD CARD
# -----------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")

pos_text = " ".join(data[data['sentiment'] == 'positive']['review'])
neg_text = " ".join(data[data['sentiment'] == 'negative']['review'])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(WordCloud(background_color="white", colormap="Greens").generate(pos_text))
ax[0].set_title("Positive Reviews"); ax[0].axis("off")
ax[1].imshow(WordCloud(background_color="white", colormap="Reds").generate(neg_text))
ax[1].set_title("Negative Reviews"); ax[1].axis("off")
st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)
