import os
import optuna
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import csv

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and process dataset
ratings_df = pd.read_csv("data/rating.csv")
ratings_df = ratings_df.dropna(subset=["user_id", "anime_id", "rating"])
ratings_df = ratings_df[ratings_df["rating"] != -1]
ratings_df["user_id"] = ratings_df["user_id"].astype(int)
ratings_df["anime_id"] = ratings_df["anime_id"].astype(int)

original_user_ids = ratings_df["user_id"].unique()
user_to_idx = {user: idx for idx, user in enumerate(original_user_ids)}
ratings_df["user_id"] = ratings_df["user_id"].map(user_to_idx)

original_anime_ids = ratings_df["anime_id"].unique()
anime_to_idx = {anime: idx for idx, anime in enumerate(original_anime_ids)}
ratings_df["anime_id"] = ratings_df["anime_id"].map(anime_to_idx)

# Split and sample train data for speed
train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_df = train_df.sample(frac=0.2, random_state=42)

class AnimeRatingsDataset(Dataset): # copied this
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.anime = torch.tensor(df["anime_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.anime[idx], self.ratings[idx]

class MatrixFactorizationModel(nn.Module): # copied this
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

def evaluate(model, val_df): # unique to this
    model.eval()
    dataset = AnimeRatingsDataset(val_df)
    dataloader = DataLoader(dataset, batch_size=128)
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for users, anime, ratings in dataloader:
            users = users.to(device)
            anime = anime.to(device)
            ratings = ratings.to(device)
            preds = model(users, anime)
            loss = criterion(preds, ratings)
            total_loss += loss.item()
    return (total_loss / len(dataloader)) ** 0.5

def objective(trial): # unique to this
    num_factors = trial.suggest_int("num_factors", 20, 100, step=10)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 5, 12)

    model = MatrixFactorizationModel(len(user_to_idx), len(anime_to_idx), num_factors).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_dataset = AnimeRatingsDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    patience = 3  # early stopping patience

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for users, anime, ratings in train_loader:
            users = users.to(device)
            anime = anime.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            preds = model(users, anime)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_rmse = evaluate(model, val_df)
        print(f"    [Trial {trial.number}] Epoch {epoch+1}/{num_epochs} - Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            # Save best model for this trial
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    [Trial {trial.number}] Early stopping at epoch {epoch+1}")
                break

    trial.set_user_attr("num_factors", num_factors)
    trial.set_user_attr("learning_rate", lr)
    trial.set_user_attr("weight_decay", weight_decay)
    trial.set_user_attr("num_epochs", num_epochs)

    return best_val_rmse

# Run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Save study results
with open("optuna_trials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "rmse", "num_factors", "learning_rate", "weight_decay", "num_epochs"])
    for i, t in enumerate(study.trials):
        writer.writerow([
            i,
            t.value,
            t.user_attrs["num_factors"],
            t.user_attrs["learning_rate"],
            t.user_attrs["weight_decay"],
            t.user_attrs["num_epochs"]
        ])

print("\\n✅ Best trial:")
print(f"  RMSE: {study.best_value:.4f}")
print(f"  Params: {study.best_params}")
print("  Model saved to best_model.pth")
