"""
clustering_demo.py — K-Means clustering demonstration for AgriSmart AI.

Uses the crop recommendation dataset features to find natural groupings.
Explains clearly that clusters are data patterns, not expert-defined categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_crop_data, CROP_FEATURE_COLS, get_crop_features

RANDOM_SEED = 42


def run_clustering(n_clusters: int = 5) -> dict:
    """
    Run K-Means clustering on crop recommendation features.

    Parameters
    ----------
    n_clusters : number of clusters (2–15 supported)

    Returns
    -------
    dict with cluster labels, PCA-reduced coordinates, inertia,
    cluster centres and the original DataFrame with cluster assignments.
    """
    n_clusters = max(2, min(15, int(n_clusters)))

    df = load_crop_data()
    X  = get_crop_features(df)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    labels = kmeans.fit_predict(X_scaled)

    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(X_scaled)

    result_df = df.copy()
    result_df["cluster"] = labels
    result_df["pca_x"]   = X_2d[:, 0]
    result_df["pca_y"]   = X_2d[:, 1]

    # Cluster centre statistics in original feature space
    centres_scaled = kmeans.cluster_centers_
    centres_original = scaler.inverse_transform(centres_scaled)
    centres_df = pd.DataFrame(centres_original, columns=CROP_FEATURE_COLS)
    centres_df.index.name = "cluster"

    return {
        "result_df":  result_df,
        "n_clusters": n_clusters,
        "inertia":    round(float(kmeans.inertia_), 2),
        "centres_df": centres_df,
        "pca_variance_explained": pca.explained_variance_ratio_.round(4),
    }


def compute_elbow(max_k: int = 10) -> dict:
    """
    Compute inertia for k=2..max_k for an elbow plot.

    Returns
    -------
    dict with k_values and inertia_values lists.
    """
    df = load_crop_data()
    X  = get_crop_features(df)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values   = list(range(2, max_k + 1))
    inertias   = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(round(float(km.inertia_), 2))

    return {"k_values": k_values, "inertia_values": inertias}
