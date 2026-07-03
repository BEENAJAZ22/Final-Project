import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from textblob import TextBlob

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="News Popularity AI", layout="centered")

st.title("🧠 News Popularity Intelligence System")
st.write("Predict how attention-grabbing a news article is.")

# -------------------------------
# Load Model (SAFE VERSION)
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    model.eval()
    return tokenizer, model


with st.spinner("Loading AI model... please wait ⏳"):
    tokenizer, model = load_model()

st.success("Model loaded successfully ✅")

# -------------------------------
# Functions
# -------------------------------
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].numpy()[0]


def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0


def urgency_score(text):
    words = ["breaking", "urgent", "crisis", "alert", "emergency"]
    text = text.lower()
    return sum(word in text for word in words) / len(words)


def sentiment(text):
    return TextBlob(text).sentiment.polarity



def emotion_intensity(text):
    text = text.lower()

    emotion_words = [
        "crisis", "disaster", "panic", "fear", "dead",
        "death", "explosion", "earthquake", "attack",
        "emergency", "war", "flood", "collapse"
    ]

    count = sum(word in text for word in emotion_words)
    return min(count / 5, 1)  # normalize


# -------------------------------
# User Input
# -------------------------------
title = st.text_input("📰 News Title")
description = st.text_area("📝 News Description")

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze 🚀"):

    if title.strip() == "" and description.strip() == "":
        st.warning("⚠ Please enter title or description")
    else:
        full_text = title + " " + description

        # Features
        length_score = min(len(full_text.split()) / 200, 1)
        diversity = lexical_diversity(full_text)
        polarity = sentiment(full_text)
        emotion=emotion = emotion_intensity(full_text)
        urgency = urgency_score(full_text)

        # Embedding
        with st.spinner("Analyzing content... 🤖"):
            embedding = get_embedding(full_text)

        embedding_score = np.linalg.norm(embedding)
        embedding_score = embedding_score / 20
        embedding_score = min(embedding_score, 1)

        # Final Score
        score = (
            0.25 * embedding_score +
            0.20 * emotion +
            0.15 * urgency +
            0.15 * diversity +
            0.10 * length_score 

        )

        # Output
        st.subheader("📊 Popularity Score")
        st.success(round(score, 3))

        # Label
        if score > 0.7:
            st.success("🔥 Highly Attention-Grabbing Article")
        elif score > 0.4:
            st.info("⚡ Moderately Popular Article")
        else:
            st.warning("😐 Low Popularity Potential")

        # Explanation
        st.subheader("🧠 Explanation")

        st.write(f"Emotion Intensity: {round(emotion, 3)}")
        st.write(f"Urgency Score: {round(urgency, 3)}")
        st.write(f"Lexical Diversity: {round(diversity, 3)}")
        st.write(f"Length Score: {round(length_score, 3)}")
        st.write(f"Embedding Score: {round(embedding_score, 3)}")