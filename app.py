# ================================
# app.py - Sentiment & Emotion Analyzer (with Dynamic Cinematic BG)
# ================================

import streamlit as st
import pandas as pd
import requests
import os
import gdown
import base64
import nltk
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------------------
# Basic page config
# -------------------------------
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")

# -------------------------------
# One-time NLTK downloads (safe on Streamlit Cloud)
# -------------------------------
nltk.download('punkt', quiet=True)
# newer textblob/nltk stacks sometimes need these:
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception:
    pass

# -------------------------------
# Dynamic Unsplash background
# -------------------------------
def fetch_unsplash_bg(query="cinema poster collage, film, movie, retro"):
    """
    Returns a high-res photo URL from Unsplash using the access key in Streamlit secrets.
    Falls back to a neutral gradient if request fails.
    """
    access_key = st.secrets.get("UNSPLASH_ACCESS_KEY", None)
    if not access_key:
        return None

    try:
        # search endpoint (1 result, landscape, safe)
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "orientation": "landscape",
            "content_filter": "high",
            "per_page": 1,
        }
        headers = {"Authorization": f"Client-ID {access_key}"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            # prefer full/regular
            photo = data["results"][0]["urls"].get("full") or data["results"][0]["urls"].get("regular")
            return photo
    except Exception:
        return None
    return None

def set_background_url(bg_url: str | None):
    """
    Inject CSS: full-page background + one glass container class (.glass-box).
    If bg_url is None, use a subtle gradient.
    """
    if bg_url:
        bg_css = f"background-image: url('{bg_url}'); background-size: cover; background-position: center; background-attachment: fixed;"
    else:
        # fallback gradient
        bg_css = "background: radial-gradient(1200px 600px at 50% -10%, #1f2937 0%, #0b0f19 60%, #000 100%);"

    css = f"""
    <style>
    /* Page background */
    [data-testid="stAppViewContainer"] {{
        {bg_css}
    }}

    /* Center column width a bit wider */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}

    /* Frosted-glass wrapper for ALL content */
    .glass-box {{
        background: rgba(0, 0, 0, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 22px;
        padding: 40px 50px;
        margin: 40px auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }}

    /* Typography contrast */
    h1, h2, h3, p, label, .stMarkdown, .stText, .stCaption, .st-subheader {{
        color: #ffffff !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }}

    /* Textarea & widgets styling for readability */
    textarea, .stTextArea textarea {{
        background: rgba(255,255,255,0.08) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }}

    /* Primary button feel */
    button[kind="primary"] {{
        background-color: #E50914 !important; /* Netflix red */
        color: #fff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.1rem !important;
        box-shadow: 0 6px 18px rgba(229,9,20,0.35);
    }}
    button[kind="primary"]:hover {{
        filter: brightness(1.08);
        transform: translateY(-1px);
    }}

    /* Wordcloud figs spacing */
    .element-container:has(canvas), .stImage, .stPlotlyChart, .stAltairChart, .stVegaLiteChart {{
        border-radius: 16px;
        overflow: hidden;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# fetch and set background once per session
if "bg_url" not in st.session_state:
    st.session_state.bg_url = fetch_unsplash_bg()
set_background_url(st.session_state.bg_url)

# -------------------------------
# Open the single glass container
# -------------------------------
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# ================================
# Header
# ================================
st.title("🎬 IMDb Movie Review - NLP Sentiment & Emotion Analyzer")
st.write("Analyze any movie review — get its **sentiment**, **dominant emotion**, and **visual insights**!")

# ================================
# Load Dataset (Google Drive via gdown)
# ================================
# Replace with your file_id if you change source
FILE_ID = "1c-6qg1kGsuDrNXS9iCH4L1ryqiARMAUM"
GD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CSV_LOCAL = "IMDB_Dataset.csv"

if not os.path.exists(CSV_LOCAL):
    try:
        gdown.download(GD_URL, CSV_LOCAL, quiet=False)
    except Exception as e:
        st.error("⚠️ Could not download dataset from Drive. Check file permissions or File ID.")
        st.stop()

try:
    data = pd.read_csv(CSV_LOCAL, on_bad_lines='skip')
except Exception as e:
    st.error("⚠️ Failed to read the CSV. Please ensure the dataset is a valid CSV file.")
    st.stop()

# Sample to keep things fast on free tiers
data = data.sample(5000, random_state=42)

# ================================
# Train classical ML model
# ================================
X = data["review"]
y = data["sentiment"]

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_tfidf, y)

# ================================
# User input
# ================================
user_input = st.text_area("✍️ Enter a movie review below:", height=150)

if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please type a review first!")
    else:
        # Sentiment prediction
        vec = tfidf.transform([user_input])
        sentiment = model.predict(vec)[0]

        # Emotion detection
        emo = NRCLex(user_input)
        scores = emo.raw_emotion_scores
        dominant_emotion = max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"

        # Results
        st.subheader("🧠 Analysis Result")
        st.write(f"**Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Dominant Emotion:** {dominant_emotion.capitalize()}")

        # Small celebration
        if sentiment.lower() == "positive":
            st.balloons()
        elif sentiment.lower() == "negative":
            st.snow()

        # Emotion breakdown
        if scores:
            st.subheader("🎭 Emotion Breakdown")
            emo_df = pd.DataFrame(scores.items(), columns=["Emotion", "Score"]).sort_values("Score", ascending=False)
            st.bar_chart(emo_df.set_index("Emotion"))
        else:
            st.info("No clear emotions detected for this text.")

        # Recommendations
        st.subheader("🍿 Movie Recommendations for You")
        recs_map = {
            "positive": [
                "The Pursuit of Happyness", "Forrest Gump", "Interstellar", "Inside Out",
                "Coco", "La La Land", "3 Idiots", "Zindagi Na Milegi Dobara",
            ],
            "negative": [
                "Joker", "The Shawshank Redemption", "Fight Club", "The Green Mile",
                "Requiem for a Dream", "Grave of the Fireflies", "Manchester by the Sea", "The Whale",
            ],
            "neutral": [
                "Inception", "Tenet", "The Prestige", "Arrival",
                "The Matrix", "Source Code", "Her", "Eternal Sunshine of the Spotless Mind",
            ],
        }
        picks = recs_map.get(sentiment.lower(), recs_map["neutral"])
        cols = st.columns(4)
        for i, mv in enumerate(picks[:8]):
            with cols[i % 4]:
                st.markdown(f"🎞️ **{mv}**")
        st.success("Hope these movies match your mood! 🍿")

# ================================
# WordCloud Visualization
# ================================
st.subheader("🌈 WordCloud of Positive vs Negative Reviews")
pos_text = " ".join(data[data["sentiment"] == "positive"]["review"])
neg_text = " ".join(data[data["sentiment"] == "negative"]["review"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(pos_text), interpolation="bilinear")
axes[0].set_title("Positive Reviews")
axes[0].axis("off")

axes[1].imshow(WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(neg_text), interpolation="bilinear")
axes[1].set_title("Negative Reviews")
axes[1].axis("off")

st.pyplot(fig)

# -------------------------------
# Close the glass container
# -------------------------------
st.markdown("</div>", unsafe_allow_html=True)
