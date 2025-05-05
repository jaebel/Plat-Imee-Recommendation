from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

app = FastAPI()

# Setup device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AnimeEntryDTO(BaseModel):
    malId: int = Field(..., alias="malId")
    rating: float = 7.0

class UserAnimeListDTO(BaseModel):
    userId: int
    animeList: List[AnimeEntryDTO]
    safeSearch: bool = False

class RecResponseDTO(BaseModel):
    mal_id: int

class AnimeRatingsDataset(Dataset):
    def __init__(self, ratings_df: pd.DataFrame):
        self.users = torch.tensor(ratings_df['user_id'].values, dtype=torch.long)
        self.anime = torch.tensor(ratings_df['anime_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.anime[idx], self.ratings[idx]

class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_anime, num_factors=30):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        self.anime_embeddings = nn.Embedding(num_anime, num_factors)
        self.user_biases = nn.Embedding(num_users, 1)
        self.anime_biases = nn.Embedding(num_anime, 1)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.anime_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.anime_biases.weight)

    def forward(self, userIds, animeIds):
        user_embeds = self.user_embeddings(userIds)
        anime_embeds = self.anime_embeddings(animeIds)
        user_bias = self.user_biases(userIds).squeeze()
        anime_bias = self.anime_biases(animeIds).squeeze()
        prediction = (user_embeds * anime_embeds).sum(dim=1) + user_bias + anime_bias
        return prediction

def train_model(ratings_df: pd.DataFrame, num_epochs=19, batch_size=128, num_factors=30, learning_rate=0.00098238482170162, weight_decay=1.7546484188145045e-06):
    dataset = AnimeRatingsDataset(ratings_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_users = ratings_df['user_id'].nunique()
    num_anime = ratings_df['anime_id'].nunique()

    model = MatrixFactorizationModel(num_users, num_anime, num_factors).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for users, anime, ratings in dataloader:
            users, anime, ratings = users.to(device), anime.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(users, anime)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

    return model

def compute_new_user_embedding(payload: UserAnimeListDTO, model) -> torch.Tensor:
    valid_indices = []
    ratings_list = []
    for entry in payload.animeList:
        if entry.malId in anime_to_idx:
            valid_indices.append(anime_to_idx[entry.malId])
            ratings_list.append(entry.rating)
    if not valid_indices:
        raise HTTPException(status_code=404, detail="No valid anime in payload found in training data")

    indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
    item_embeddings = model.anime_embeddings.weight.data[indices_tensor]
    weights_tensor = torch.tensor(ratings_list, dtype=torch.float32, device=device).unsqueeze(1)
    user_embedding = (item_embeddings * weights_tensor).sum(dim=0) / weights_tensor.sum()
    return user_embedding

def recommend_anime_for_user(user_id: int, payload: UserAnimeListDTO, top_n=10):
    if user_id < 1_000_000:
        user_id += 1_000_000

    if user_id in user_to_idx:
        user_idx = user_to_idx[user_id]
        num_anime = len(unique_anime_rev)
        all_anime_indices = torch.arange(num_anime, dtype=torch.long, device=device)
        user_tensor = torch.tensor([user_idx] * num_anime, dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            predictions = model(user_tensor, all_anime_indices).cpu().numpy()
    else:
        new_user_embedding = compute_new_user_embedding(payload, model).to(device)
        all_item_embeddings = model.anime_embeddings.weight.data
        all_item_biases = model.anime_biases.weight.data.squeeze()
        predictions = (all_item_embeddings @ new_user_embedding) + all_item_biases
        predictions = predictions.cpu().numpy()

    user_anime_ids = {entry.malId for entry in payload.animeList}

    filtered = []
    for idx, score in enumerate(predictions):
        mal_id = unique_anime_rev[idx]
        if mal_id in user_anime_ids:
            continue
        genres = anime_genres.get(mal_id, "").lower()
        if payload.safeSearch and ("hentai" in genres or "ecchi" in genres):
            continue
        filtered.append((idx, score))

    top_indices = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]
    rec_anime_ids = [unique_anime_rev[idx] for idx, _ in top_indices]
    return rec_anime_ids

model = None
ratings_df = None
user_to_idx = {}
anime_to_idx = {}
unique_anime_rev = None
anime_genres = {}

@app.on_event("startup")
def startup_event():
    global model, ratings_df, user_to_idx, anime_to_idx, unique_anime_rev, anime_genres
    try:
        ratings_df = pd.read_csv("data/rating.csv")
        ratings_df = ratings_df.dropna(subset=['user_id', 'anime_id', 'rating'])
        ratings_df = ratings_df[ratings_df['rating'] != -1]
        ratings_df['user_id'] = ratings_df['user_id'].astype(int)
        ratings_df['anime_id'] = ratings_df['anime_id'].astype(int)

        original_user_ids = ratings_df['user_id'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(original_user_ids)}
        ratings_df['user_id'] = ratings_df['user_id'].map(user_to_idx)

        original_anime_ids = ratings_df['anime_id'].unique()
        anime_to_idx = {anime: idx for idx, anime in enumerate(original_anime_ids)}
        ratings_df['anime_id'] = ratings_df['anime_id'].map(anime_to_idx)
        unique_anime_rev = list(original_anime_ids)

        anime_df = pd.read_csv("data/anime.csv")
        required_columns = ['MAL_ID', 'Genres']
        for col in required_columns:
            if col not in anime_df.columns:
                raise ValueError(f"Column '{col}' not found in anime.csv.")
        anime_df = anime_df.dropna(subset=required_columns)
        anime_df['MAL_ID'] = anime_df['MAL_ID'].astype(int)

        for _, row in anime_df.iterrows():
            anime_genres[row['MAL_ID']] = row['Genres']

        model = MatrixFactorizationModel(len(user_to_idx), len(anime_to_idx)).to(device)

        if os.path.exists("anime_recommender.pth"):
            print("Loading saved model...")
            model.load_state_dict(torch.load("anime_recommender.pth", map_location=device))
            model.eval()
        else:
            print("Training model on startup...")
            model = train_model(
                ratings_df,
                num_epochs=19,
                num_factors=30,
                learning_rate=0.00098238482170162,
                weight_decay=1.7546484188145045e-06
            )
            torch.save(model.state_dict(), "anime_recommender.pth")
            print("Model training completed and saved.")

    except Exception as e:
        print("Error during startup training:", e)

@app.post("/api/recommendations", response_model=List[RecResponseDTO])
def get_recommendations(payload: UserAnimeListDTO):
    print(f"[INFO-LOG] Received recommendation request: userId={payload.userId}, safeSearch={payload.safeSearch}, animeList={[entry.dict() for entry in payload.animeList]}")
    try:
        rec_anime_ids = recommend_anime_for_user(payload.userId, payload, top_n=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [RecResponseDTO(mal_id=anime_id) for anime_id in rec_anime_ids]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=5000, reload=True)

def get_model_resources():
    return model, anime_to_idx, unique_anime_rev, anime_genres
