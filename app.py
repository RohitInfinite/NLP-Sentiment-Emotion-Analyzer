import streamlit as st
import pandas as pd
import base64
import os
import gdown
import nltk
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================
# MUST BE FIRST
# ============================
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")

# ============================
# Background Image Function
# ============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    encoded = base64.b64encode(img_data).decode()

    st.markdown("""
    <style>
    .glass-box {
        background: rgba(0, 0, 0, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 40px 50px;
        margin: 50px auto;
        max-width: 1100px;
        box-shadow: 0 8px 50px rgba(0,0,0,0.7);
    }

    h1, h2, h3, p, label, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.9);
    }

    .stTextInput textarea, .stTextArea textarea {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }

    .stButton>button {
        background-color: #ff4747 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600;
        padding: 10px 22px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7474 !important;
        transform: scale(1.05);
    }
    </style>

    <div class="glass-box">
""", unsafe_allow_html=True)
    
set_background("samuel-regan-asante-wMkaMXTJjlQ-unsplash.jpg")  # << CHANGE ONLY IF YOUR IMAGE NAME IS DIFFERENT

# ============================
# Glass Container Start
# ============================
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

# ============================
# Title
# ============================
st.title("🎬 IMDb Movie Review - NLP Sentiment & Emotion Analyzer")
st.write("Analyze any movie review and discover its **sentiment**, **dominant emotion**, and **visual insights**!")

# ============================
# Download Required NLTK
# ============================
nltk.download('punkt', quiet=True)

# ============================
# Dataset Load (Google Drive Auto Download)
# ============================
file_id = "1c-6qg1kGsuDrNXS9iCH4L1ryqiARMAUM"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("IMDB_Dataset.csv"):
    gdown.download(url, "IMDB_Dataset.csv", quiet=False)

data = pd.read_csv("IMDB_Dataset.csv", on_bad_lines='skip').sample(5000, random_state=42)

X = data['review']
y = data['sentiment']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# ============================
# Input Area
# ============================
review = st.text_area("✍️ Enter a movie review:", height=150)

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment = model.predict(tfidf.transform([review]))[0]

        emo = NRCLex(review)
        scores = emo.raw_emotion_scores
        dominant = max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"

        st.subheader("🧠 Analysis Result")
        st.write(f"**Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Dominant Emotion:** {dominant.capitalize()}")

        if scores:
            st.subheader("🎭 Emotion Breakdown")
            emo_df = pd.DataFrame(scores.items(), columns=['Emotion', 'Score'])
            emo_df = emo_df.sort_values(by='Score', ascending=False)
            st.bar_chart(emo_df.set_index('Emotion'))

        # Movie Recommendations
        st.subheader("🍿 Movie Recommendations Based on Mood")
        movies = {
            "positive": ["The Pursuit of Happyness", "Forrest Gump", "Interstellar", "Inside Out", "Coco", "La La Land", "3 Idiots", "Zindagi Na Milegi Dobara"],
            "negative": ["Joker", "The Green Mile", "The Whale", "Grave of the Fireflies", "Fight Club", "Manchester by the Sea", "Requiem for a Dream"],
            "neutral": ["Inception", "Tenet", "Arrival", "The Matrix", "The Prestige", "Source Code", "Her"]
        }
        for m in movies.get(sentiment.lower(), movies["neutral"]):
            st.markdown(f"🎞️ **{m}**")

        st.success("Enjoy watching 🎥")

# ============================
# WordCloud
# ============================
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")
pos_text = " ".join(data[data['sentiment']=="positive"]['review'])
neg_text = " ".join(data[data['sentiment']=="negative"]['review'])

fig, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].imshow(WordCloud(background_color='white', colormap='Greens').generate(pos_text))
ax[0].set_title("Positive Reviews"); ax[0].axis("off")
ax[1].imshow(WordCloud(background_color='white', colormap='Reds').generate(neg_text))
ax[1].set_title("Negative Reviews"); ax[1].axis("off")
st.pyplot(fig)

# ============================
# Close Glass Container
# ============================
st.markdown("</div>", unsafe_allow_html=True)
