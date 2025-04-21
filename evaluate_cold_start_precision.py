import pandas as pd
import torch
from typing import List
from main import (
    MatrixFactorizationModel,
    compute_new_user_embedding,
    recommend_anime_for_user,
    AnimeEntryDTO,
    UserAnimeListDTO,
    startup_event,
    get_model_resources
)

# Initialize model and resources
startup_event()
model, anime_to_idx, unique_anime_rev, anime_genres = get_model_resources()

# Load and clean ratings data
ratings_df = pd.read_csv("data/rating.csv")
ratings_df = ratings_df.dropna(subset=["user_id", "anime_id", "rating"])
ratings_df = ratings_df[ratings_df["rating"] != -1]
ratings_df["user_id"] = ratings_df["user_id"].astype(int)
ratings_df["anime_id"] = ratings_df["anime_id"].astype(int)

# Config
K = 10
min_relevant_rating = 7.0

# Bucket users by rating count
user_groups = ratings_df.groupby("user_id")
buckets = {
    "1-2": [],
    "3-5": [],
    "6-10": [],
    "11+": []
}

for user_id, group in user_groups:
    num_ratings = len(group)
    if num_ratings <= 2:
        buckets["1-2"].append((user_id, group))
    elif num_ratings <= 5:
        buckets["3-5"].append((user_id, group))
    elif num_ratings <= 10:
        buckets["6-10"].append((user_id, group))
    else:
        buckets["11+"].append((user_id, group))

# Evaluate each bucket
results = []

for bucket_name, user_data in buckets.items():
    hit_count = 0
    precision_total = 0
    total_users = 0
    skipped = 0

    for user_id, group in user_data:
        group_sorted = group.sort_values("rating", ascending=False)
        test_row = group_sorted.iloc[0]

        if test_row["rating"] < min_relevant_rating:
            skipped += 1
            continue

        train_rows = group_sorted.iloc[1:]

        anime_list = [
            AnimeEntryDTO(malId=row["anime_id"], rating=row["rating"])
            for _, row in train_rows.iterrows()
            if row["anime_id"] in unique_anime_rev
        ]

        if len(anime_list) < 1 or test_row["anime_id"] not in unique_anime_rev:
            continue

        payload = UserAnimeListDTO(userId=user_id, animeList=anime_list, safeSearch=False)

        try:
            recommended_ids = recommend_anime_for_user(user_id, payload, top_n=K)
        except Exception as e:
            continue

        is_hit = test_row["anime_id"] in recommended_ids
        hit_count += int(is_hit)
        precision_total += (1 if is_hit else 0) / K
        total_users += 1

    if total_users > 0:
        results.append({
            "Bucket": bucket_name,
            "Users_Evaluated": total_users,
            "Skipped": skipped,
            "Precision@10": round(precision_total / total_users, 4),
            "HitRate@10": round(hit_count / total_users, 4)
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("cold_start_bucket_results.csv", index=False)

# Display bucketed results
print("\n--- Evaluation Complete ---")
print(results_df)

# Print overall weighted Precision@10
overall = results_df[["Users_Evaluated", "Precision@10"]]
weighted_avg = (overall["Users_Evaluated"] * overall["Precision@10"]).sum() / overall["Users_Evaluated"].sum()
print(f"\nOverall weighted Precision@10 across all buckets: {weighted_avg:.4f}")
