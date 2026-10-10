"""
Generate sample crop recommendation dataset for AgriSmart AI.
This script creates a synthetic dataset that mirrors the structure of the
publicly available Crop Recommendation Dataset (Kaggle / UCI).

NOTE: This is sample data for demonstration purposes.
For best results, replace with the actual dataset from:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Crop profiles: (N_mean, P_mean, K_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean)
CROP_PROFILES = {
    "rice":        (80, 40, 40, 24, 82, 6.5, 200),
    "maize":       (70, 45, 35, 22, 65, 6.0, 80),
    "chickpea":    (40, 70, 80, 18, 20, 7.2, 70),
    "kidneybeans": (20, 70, 20, 20, 22, 5.9, 100),
    "pigeonpeas":  (20, 68, 20, 27, 49, 5.8, 150),
    "mothbeans":   (20, 48, 20, 28, 53, 6.7, 55),
    "mungbean":    (20, 48, 20, 28, 85, 6.7, 45),
    "blackgram":   (40, 68, 20, 30, 66, 7.1, 68),
    "lentil":      (18, 68, 20, 24, 65, 6.9, 45),
    "pomegranate": (18, 18, 40, 22, 90, 6.0, 110),
    "banana":      (100, 82, 50, 27, 80, 5.8, 105),
    "mango":       (20, 27, 30, 31, 50, 5.8, 95),
    "grapes":      (23, 132, 200, 24, 82, 6.0, 70),
    "watermelon":  (99, 17, 50, 25, 85, 6.5, 50),
    "muskmelon":   (100, 17, 50, 29, 93, 6.3, 25),
    "apple":       (20, 134, 200, 22, 92, 6.0, 113),
    "orange":      (20, 16, 10, 23, 92, 7.0, 110),
    "papaya":      (49, 59, 50, 34, 92, 6.7, 145),
    "coconut":     (22, 16, 30, 27, 94, 5.9, 175),
    "cotton":      (118, 46, 20, 24, 80, 7.0, 80),
    "jute":        (78, 46, 40, 25, 80, 6.7, 175),
    "coffee":      (101, 28, 30, 25, 58, 6.6, 155),
}

SAMPLES_PER_CROP = 100


def generate_dataset() -> pd.DataFrame:
    rows = []
    for crop, (n, p, k, t, h, ph, r) in CROP_PROFILES.items():
        for _ in range(SAMPLES_PER_CROP):
            rows.append({
                "N":           max(0, rng.normal(n,   n * 0.12)),
                "P":           max(0, rng.normal(p,   p * 0.12)),
                "K":           max(0, rng.normal(k,   k * 0.12)),
                "temperature": max(5, rng.normal(t,   t * 0.06)),
                "humidity":    max(5, min(100, rng.normal(h,  h * 0.07))),
                "ph":          max(3, min(10,  rng.normal(ph, 0.3))),
                "rainfall":    max(10, rng.normal(r,  r * 0.15)),
                "label":       crop,
            })
    df = pd.DataFrame(rows)
    # Round numeric columns to 2 decimal places
    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    df[num_cols] = df[num_cols].round(2)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(__file__), "crop_recommendation.csv")
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path}  ({len(df)} rows, {df['label'].nunique()} crops)")
