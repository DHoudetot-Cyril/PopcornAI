# app.py
import json
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from tmdbv3api import TMDb, Movie
from PIL import Image, ImageDraw, ImageFont

# ---- TMDb setup ----
tmdb = TMDb()
tmdb.api_key = "9aa945853994178d69aeb6717fe7a128"  # <-- remplace par ta clé TMDb
tmdb.language = 'fr-FR'
movie_api = Movie()

def get_poster(title: str):
    try:
        results = movie_api.search(title)
        if not results:
            return None

        first = results[0]
        if not isinstance(first, dict):
            try:
                first = first.__dict__
            except Exception:
                return None

        poster_path = first.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except Exception as e:
        print(f"[TMDb ERROR] {title} → {e}")
    return None

# ---- Placeholder image ----
def placeholder_image(text="Image indisponible", size=(200, 300), bg_color="lightgrey"):
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)  # police lisible
    except:
        font = ImageFont.load_default()

    # Nouvelle façon de calculer la taille du texte
    bbox = draw.textbbox((0, 0), text, font=font)  # (x0, y0, x1, y1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Centrer le texte
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2

    draw.text((x, y), text, fill="black", font=font)
    return img

def get_poster_or_placeholder(title: str):
    url = get_poster(title)
    if url:
        return url
    return placeholder_image()

# ---- Modèle MF ----
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias    = nn.Embedding(n_users, 1)
        self.item_bias    = nn.Embedding(n_items, 1)

    def forward(self, users, items):
        u = self.user_factors(users)
        v = self.item_factors(items)
        dot = (u * v).sum(dim=1, keepdim=True)
        out = dot + self.user_bias(users) + self.item_bias(items)
        return out.squeeze(1)

# ---- Chargement des artefacts ----
@st.cache_resource
def load_artifacts(artifacts_dir: str):
    artifacts = Path(artifacts_dir)
    with open(artifacts / "config.json") as f:
        config = json.load(f)
    with open(artifacts / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    ckpt = torch.load(artifacts / "model.pt", map_location="cpu")
    model = MatrixFactorization(
        n_users=ckpt["n_users"],
        n_items=ckpt["n_items"],
        n_factors=ckpt["factors"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, mappings, config

@st.cache_data
def load_movies(movies_csv: str):
    movies = pd.read_csv(movies_csv)  # movieId,title,genres
    return movies

def scores_for_user(model, user_index: int, n_items: int, device="cpu"):
    user_tensor = torch.full((n_items,), user_index, dtype=torch.long, device=device)
    item_tensor = torch.arange(n_items, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(user_tensor, item_tensor).clamp(0.5, 5.0).cpu().numpy()
    return scores

def main():
    st.set_page_config(page_title="PopcornAI 🎬", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🍿 PopcornAI — Recommandations de films")

    st.sidebar.header("Chargement")
    artifacts_dir = st.sidebar.text_input("Dossier artefacts", value="artifacts")
    movies_csv    = st.sidebar.text_input("movies.csv", value="./ml-32m/movies.csv")

    if not Path(artifacts_dir, "model.pt").exists():
        st.warning("Aucun modèle trouvé. Lance d'abord `python train.py`.")
        st.stop()

    model, mappings, config = load_artifacts(artifacts_dir)
    movies = load_movies(movies_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"🖥️ Device: **{device}**")
    model.to(device)

    user_ids = mappings["user_ids"]
    item_ids = mappings["item_ids"]

    st.sidebar.header("Utilisateur")
    default_userid = int(user_ids[0]) if len(user_ids) > 0 else 1
    user_id_input = st.sidebar.number_input(
        "userId (MovieLens)", min_value=int(user_ids.min()),
        max_value=int(user_ids.max()), value=default_userid, step=1
    )

    @st.cache_data
    def build_user_index(user_ids_array):
        return {int(uid): i for i, uid in enumerate(user_ids_array)}
    @st.cache_data
    def build_item_index(item_ids_array):
        return {int(mid): i for i, mid in enumerate(item_ids_array)}

    user_index_map = build_user_index(user_ids)
    item_index_map = build_item_index(item_ids)

    if int(user_id_input) not in user_index_map:
        st.error("Cet userId n'existe pas dans le modèle.")
        st.stop()

    user_idx = user_index_map[int(user_id_input)]
    n_items = len(item_ids)

    st.sidebar.header("Options")
    top_k = st.sidebar.slider("Top-N recommandations", 5, 100, 20, 5)
    hide_seen = st.sidebar.checkbox("Masquer les films déjà notés", value=True)

    ratings_user_path = st.sidebar.text_input("ratings_user.csv (optionnel pour masquer vus)", value="")

    seen_item_idx = set()
    if hide_seen and ratings_user_path and Path(ratings_user_path).exists():
        ru = pd.read_csv(ratings_user_path, usecols=["userId","movieId"])
        ru = ru[ru["userId"] == int(user_id_input)]
        seen_movie_ids = ru["movieId"].astype(int).tolist()
        seen_item_idx = {item_index_map[mid] for mid in seen_movie_ids if int(mid) in item_index_map}

    # ---- FILTRE PAR GENRES ----
    all_genres = set(g for gs in movies['genres'].dropna() for g in gs.split('|'))
    selected_genres = st.sidebar.multiselect("Filtrer par genres", sorted(all_genres))

    # ---- RECHERCHE PAR TITRE ----
    search_title = st.sidebar.text_input("Rechercher un film par titre")

    st.write("### 🎯 Recommandations personnalisées")
    scores = scores_for_user(model, user_idx, n_items, device=device)
    if hide_seen and len(seen_item_idx) > 0:
        scores_masked = scores.copy()
        scores_masked[list(seen_item_idx)] = -1e9
        idx_sorted = np.argsort(-scores_masked)[:top_k]
    else:
        idx_sorted = np.argsort(-scores)[:top_k]

    top_movies = pd.DataFrame({
        "movieId": item_ids[idx_sorted].astype(int),
        "pred_score": scores[idx_sorted]
    }).merge(movies, on="movieId", how="left")

    if selected_genres:
        top_movies = top_movies[top_movies['genres'].apply(lambda g: any(gen in g.split('|') for gen in selected_genres))]

    # ---- FILM ALEATOIRE ----
    st.write("### 🎲 Film aléatoire qui pourrait te plaire")
    random_movies = movies.sample(10)
    cols = st.columns(5)
    for i, row in random_movies.iterrows():
        with cols[i%5]:
            st.image(get_poster_or_placeholder(row['title']))
            st.markdown(f"**{row['title']}**")
            st.markdown(f"*{row['genres']}*")

    # ---- RECHERCHE TITRE ----
    if search_title:
        st.write(f"### 🔍 Résultats pour : {search_title}")
        search_movies = movies[movies['title'].str.contains(search_title, case=False, regex=False)]
        cols = st.columns(5)
        for i, row in search_movies.iterrows():
            with cols[i%5]:
                st.image(get_poster_or_placeholder(row['title']))
                st.markdown(f"**{row['title']}**")
                st.markdown(f"*{row['genres']}*")

    # ---- GRID DES RECOMMANDATIONS ----
    st.write("### 🌟 Top recommandations")
    cols = st.columns(5)
    for i, row in top_movies.iterrows():
        with cols[i%5]:
            st.image(get_poster_or_placeholder(row['title']))
            st.markdown(f"**{row['title']}**")
            st.markdown(f"*{row['genres']}*")
            st.markdown(f"⭐ {row['pred_score']:.2f}")

    st.caption("Modèle : Matrix Factorization (embeddings). Scores = note prédite (0.5–5.0).")

if __name__ == "__main__":
    main()
