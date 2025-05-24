import pandas as pd
import json

# Load your training CSVs
ratings_df = pd.read_csv("data/rating.csv")
anime_df = pd.read_csv("data/anime.csv")

# Process ratings to create mappings
ratings_df = ratings_df.dropna(subset=['user_id', 'anime_id', 'rating'])
ratings_df = ratings_df[ratings_df['rating'] != -1]
ratings_df['user_id'] = ratings_df['user_id'].astype(int)
ratings_df['anime_id'] = ratings_df['anime_id'].astype(int)

original_user_ids = ratings_df['user_id'].unique()
original_anime_ids = ratings_df['anime_id'].unique()

user_to_idx = {int(user): int(idx) for idx, user in enumerate(original_user_ids)}
anime_to_idx = {int(anime): int(idx) for idx, anime in enumerate(original_anime_ids)}
unique_anime_rev = [int(aid) for aid in original_anime_ids]

# Genre info for safe search
anime_df = anime_df[['MAL_ID', 'Genres']].dropna()
anime_df['MAL_ID'] = anime_df['MAL_ID'].astype(int)
anime_genres = {int(row['MAL_ID']): row['Genres'] for _, row in anime_df.iterrows()}

# Save everything
with open("anime_to_idx.json", "w") as f:
    json.dump(anime_to_idx, f)

with open("unique_anime_rev.json", "w") as f:
    json.dump(unique_anime_rev, f)

with open("anime_genres.json", "w") as f:
    json.dump(anime_genres, f)

print("✅ Export complete.")
print(f"Users: {len(user_to_idx)}, Anime: {len(anime_to_idx)}")
