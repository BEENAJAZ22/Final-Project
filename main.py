import pandas as pd

# Load dataset
df = pd.read_csv("data/News_dataset.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Remove missing values
df = df.dropna(subset=["title", "description"])

# Combine title + description
df["full_text"] = df["title"].str.strip() + " " + df["description"].str.strip()

print("Data loaded:", df.shape)

#sample subset
df_small = df.sample(5000, random_state=42).reset_index(drop=True)

print("Using subset:", df_small.shape)

from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cpu")

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

model.to(device)
model.eval()

print("Model loaded!")

#convert text to embedding
from tqdm import tqdm

def get_embeddings(text_list, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls)

    return torch.cat(all_embeddings)

embeddings = get_embeddings(df_small["full_text"].tolist())

print("Embedding shape:", embeddings.shape)

#Create Features 
#length score
df_small["length_score"] = df_small["full_text"].apply(lambda x: len(x.split()))
df_small["length_score"] = df_small["length_score"] / df_small["length_score"].max()

#Lexical Diversity
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

df_small["lexical_diversity"] = df_small["full_text"].apply(lexical_diversity)

#Emotion (Sentiment)
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df_small["polarity"] = df_small["full_text"].apply(get_sentiment)

df_small["emotion_intensity"] = df_small["polarity"].abs()

#Urgency Score
urgent_words = ["breaking", "urgent", "crisis", "alert", "emergency"]

def urgency(text):
    text = text.lower()
    return sum(word in text for word in urgent_words) / len(urgent_words)

df_small["urgency_score"] = df_small["full_text"].apply(urgency)

#Popularity Score
df_small["popularity_score"] = (
    0.3 * df_small["emotion_intensity"] +
    0.2 * df_small["urgency_score"] +
    0.2 * df_small["lexical_diversity"] +
    0.3 * df_small["length_score"]
)

#Ranking
df_small = df_small.sort_values(by="popularity_score", ascending=False)

print("\nTop 5 Articles:\n")
print(df_small[["title", "popularity_score"]].head())

#adding embeddings

import numpy as np

# Convert embeddings to numpy
embeddings_np = embeddings.numpy()

# Calculate magnitude (importance of text)
embedding_norm = np.linalg.norm(embeddings_np, axis=1)

# Normalize
embedding_norm = embedding_norm / embedding_norm.max()

# Add to dataframe
df_small["embedding_score"] = embedding_norm

#Updating Final Score

df_small["popularity_score"] = (
    0.25 * df_small["embedding_score"] +   # NEW
    0.20 * df_small["emotion_intensity"] +
    0.15 * df_small["urgency_score"] +
    0.15 * df_small["lexical_diversity"] +
    0.10 * df_small["length_score"] +
    0.15 * df_small["polarity"].abs()
)

#Re-rank
df_small = df_small.sort_values(by="popularity_score", ascending=False)

print("\nUpdated Top 5 Articles:\n")
print(df_small[["title", "popularity_score"]].head())

#explanation Function
def explain_article(row):
    return {
        "emotion": float(row["emotion_intensity"]),
        "urgency": float(row["urgency_score"]),
        "diversity": float(row["lexical_diversity"]),
        "length": float(row["length_score"]),
        "embedding": float(row["embedding_score"])
    }
    


top_article = df_small.iloc[0]

print("\nTop Article:\n", top_article["title"])

print("\nWhy it is popular:\n")
print(explain_article(top_article))