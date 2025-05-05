import os
import optuna
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import csv

# [New] GPU detection only in tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# [Copied from main.py] Load and clean ratings data
ratings_df = pd.read_csv("data/rating.csv")
ratings_df = ratings_df.dropna(subset=["user_id", "anime_id", "rating"])
ratings_df = ratings_df[ratings_df["rating"] != -1]
ratings_df["user_id"] = ratings_df["user_id"].astype(int)
ratings_df["anime_id"] = ratings_df["anime_id"].astype(int)

# [Copied from main.py] Map user/anime IDs to index space
original_user_ids = ratings_df["user_id"].unique()
user_to_idx = {user: idx for idx, user in enumerate(original_user_ids)}
ratings_df["user_id"] = ratings_df["user_id"].map(user_to_idx)

original_anime_ids = ratings_df["anime_id"].unique()
anime_to_idx = {anime: idx for idx, anime in enumerate(original_anime_ids)}
ratings_df["anime_id"] = ratings_df["anime_id"].map(anime_to_idx)

# [New] Create train/val split and sample for speed
train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
# train_df = train_df.sample(frac=0.2, random_state=42)  # speed-up trick disabled for full training set

# [Copied from main.py] Dataset class
class AnimeRatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.anime = torch.tensor(df["anime_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.anime[idx], self.ratings[idx]

# [Copied from main.py] Matrix Factorization model
class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_anime, num_factors):
        super().__init__()
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
        return (user_embeds * anime_embeds).sum(dim=1) + user_bias + anime_bias

# [New] Evaluation on validation set
def evaluate(model, val_df):
    model.eval()
    dataset = AnimeRatingsDataset(val_df)
    dataloader = DataLoader(dataset, batch_size=128)
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for users, anime, ratings in dataloader:
            users, anime, ratings = users.to(device), anime.to(device), ratings.to(device)
            preds = model(users, anime)
            loss = criterion(preds, ratings)
            total_loss += loss.item() * len(ratings)
    return total_loss / len(val_df)  # MSE

# [New] Tuning objective function

def objective(trial):
    # Expanded search space for better exploration
    num_factors = trial.suggest_int("num_factors", 20, 120, step=10)  # was 20–100
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)    # was 1e-4 to 5e-3
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    num_epochs = trial.suggest_int("num_epochs", 8, 20)  # was 5–12

    model = MatrixFactorizationModel(len(user_to_idx), len(anime_to_idx), num_factors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_dataset = AnimeRatingsDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_val_mse = float("inf")
    patience = 3
    num_bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        for users, anime, ratings in train_loader:
            users, anime, ratings = users.to(device), anime.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(users, anime)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

        val_mse = evaluate(model, val_df)
        print(f"    [Trial {trial.number}] Epoch {epoch+1}/{num_epochs} - Val MSE: {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs >= patience:
                print(f"    [Trial {trial.number}] Early stopping at epoch {epoch+1}")
                break

    return best_val_mse

# [New] Run Optuna Study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# [New] Save best model and trial info
best_trial = study.best_trial
print("\n Best trial:")
print(f"  MSE: {best_trial.value:.4f}")
print(f"  Params: {best_trial.params}")

# Save model with best config
best_model = MatrixFactorizationModel(len(user_to_idx), len(anime_to_idx), best_trial.params['num_factors']).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params['learning_rate'], weight_decay=best_trial.params['weight_decay'])
train_dataset = AnimeRatingsDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
criterion = nn.MSELoss()

for epoch in range(best_trial.params['num_epochs']):
    best_model.train()
    for users, anime, ratings in train_loader:
        users, anime, ratings = users.to(device), anime.to(device), ratings.to(device)
        optimizer.zero_grad()
        preds = best_model(users, anime)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()

torch.save(best_model.state_dict(), "best_model.pth")

# Save trial results
with open("optuna_trials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "mse"] + list(best_trial.params.keys()))
    for i, t in enumerate(study.trials):
        writer.writerow([i, t.value] + [t.params.get(k, None) for k in best_trial.params.keys()])
