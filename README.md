# Plat-Imee — Recommendation Microservice

A standalone Python microservice that generates personalised anime recommendations using a **neural network-based matrix factorisation model** built with PyTorch. It is called internally by the Spring Boot backend and operates independently of the main web stack.

📄 **[Read the Full Project Report](./Plat-Imee_Final_Report.docx)**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Primary language |
| FastAPI | Web framework for the recommendation endpoint |
| Uvicorn | ASGI server |
| PyTorch | Matrix factorisation model implementation |
| Pydantic | Request/response validation |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Optuna | Hyperparameter tuning |

---

## Project Structure

```
recommendation-platimee/
├── main.py                          # FastAPI app, model loading, inference logic
├── optuna_tuner.py                  # Hyperparameter optimisation script
├── evaluate_cold_start_precision.py # Model evaluation (Precision@10, HitRate@10)
├── anime_recommender.pth            # Trained model weights (generated on first run)
├── best_model.pth                   # Best model weights from Optuna tuning
├── anime_to_idx.json                # Anime ID to index mapping
├── unique_anime_rev.json            # Reverse index to anime ID mapping
├── anime_genres.json                # Anime genre data for safe search filtering
├── optuna_trials.csv                # Results from hyperparameter tuning trials
├── cold_start_bucket_results.csv    # Cold-start evaluation output
├── data/
│   ├── anime.csv                    # Anime metadata (genres, titles, MAL IDs)
│   └── rating.csv                   # Historical user-anime ratings for training
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `anime.csv` and `rating.csv` placed in the `data/` directory

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Plat-Imee-Recommendation.git
   cd recommendation-platimee
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the service:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

   The service runs on `http://localhost:8001` by default.

> **Note:** On startup, `main.py` automatically trains the matrix factorisation model from `rating.csv` if no saved weights are found, then loads it into memory ready to serve requests.

---

## API

### `POST /api/recommendations`

Accepts a user's watchlist and returns a ranked list of recommended anime MAL IDs.

**Request body:**
```json
{
  "userId": 1000001,
  "animeList": [
    { "malId": 20, "rating": 9.0 },
    { "malId": 1535, "rating": 8.0 }
  ],
  "safeSearch": true
}
```

**Response:**
```json
{
  "recommendations": [16498, 5114, 9253, 11061]
}
```

- Anime the user has already rated are automatically excluded from results.
- If `safeSearch` is `true`, anime tagged as `ecchi` or `hentai` are filtered out.

---

## How the Model Works

### Matrix Factorisation

The model is a shallow neural network using PyTorch embedding layers. It learns:

- A **user embedding** — a dense vector capturing latent preferences
- An **anime embedding** — a dense vector capturing latent content characteristics
- A **user bias** — accounts for consistently lenient or harsh raters
- An **anime bias** — accounts for consistently high or low-rated titles

The predicted affinity score for a (user, anime) pair is:

```
score = dot(user_embedding, anime_embedding) + user_bias + anime_bias
```

Training minimises **Mean Squared Error (MSE)** using the **Adam optimiser**.

### Cold-Start Handling

Since the model is not retrained after deployment, all Plat-Imee users are treated as cold-start users. A temporary user embedding is constructed at inference time by averaging the embeddings of their rated anime, weighted by their ratings:

```
user_embedding = mean(rating_i * anime_embedding_i  for each rated anime i)
```

This allows meaningful recommendations even for brand-new users with only a handful of ratings.

---

## Hyperparameter Tuning

Run `optuna_tuner.py` to search for the best model configuration over 30 trials using an 80/20 train/validation split:

```bash
python optuna_tuner.py
```

Results are saved to `optuna_trials.csv`. The best configuration found:

| Hyperparameter | Value |
|---|---|
| Latent factors | 30 |
| Learning rate | 0.000982 |
| Weight decay | 1.75e-06 |
| Epochs | 19 |
| Validation MSE | **1.4606** |

---

## Model Evaluation

Run the cold-start evaluation script to measure **Precision@10** and **HitRate@10** across user groups:

```bash
python evaluate_cold_start_precision.py
```

Results are written to `cold_start_bucket_results.csv`.

| User Group (# ratings) | HitRate@10 | Precision@10 |
|---|---|---|
| 1–2 | 24.4% | 0.0244 |
| 3–5 | 31.6% | 0.0316 |
| 6–10 | **33.4%** | **0.0334** |
| 11+ | 30.1% | 0.0301 |

The evaluation simulates cold-start conditions by withholding each user's highest-rated anime and checking whether it appears in the top 10 recommendations.
