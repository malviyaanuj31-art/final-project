"""
page_explore.py — Explore Agricultural Data page for AgriSmart AI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def show() -> None:
    st.title("📊 Explore Agricultural Data")
    st.markdown(
        "This page lets you explore the crop recommendation dataset that AgriSmart AI "
        "uses to train its models. Browse the data, check distributions, and understand "
        "the relationship between soil conditions and crop categories."
    )

    # ── load data ────────────────────────────────────────────────────────────────
    try:
        from src.data_loader import load_crop_data, CROP_FEATURE_COLS
        df = load_crop_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    # ── dataset overview ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📋 Dataset Overview</p>', unsafe_allow_html=True)

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Rows", f"{len(df):,}")
    ov2.metric("Columns", len(df.columns))
    ov3.metric("Crop Categories", df["label"].nunique())
    ov4.metric("Missing Values", int(df.isnull().sum().sum()))

    # ── data preview ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔍 Data Preview</p>', unsafe_allow_html=True)

    n_rows = st.slider("Number of rows to show", 5, 50, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    with st.expander("Column names and data types"):
        type_df = pd.DataFrame({
            "Column":    df.columns.tolist(),
            "Data Type": df.dtypes.astype(str).tolist(),
        })
        st.dataframe(type_df, use_container_width=True)

    with st.expander("Missing-value summary"):
        miss = df.isnull().sum()
        miss_df = pd.DataFrame({"Column": miss.index, "Missing": miss.values})
        if miss.sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            st.dataframe(miss_df[miss_df["Missing"] > 0], use_container_width=True)

    # ── summary statistics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📈 Summary Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df[CROP_FEATURE_COLS].describe().round(2), use_container_width=True)

    # ── crop distribution ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🌾 Crop Category Distribution</p>', unsafe_allow_html=True)

    crop_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(crop_counts.index, crop_counts.values, color="#16a34a", edgecolor="white")
    ax.set_xlabel("Number of Records")
    ax.set_title("Crop Distribution in Dataset")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── feature histograms ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Feature Distributions</p>', unsafe_allow_html=True)

    selected_features = st.multiselect(
        "Select features to plot",
        CROP_FEATURE_COLS,
        default=CROP_FEATURE_COLS,
    )

    if selected_features:
        n_cols = 3
        n_rows_fig = -(-len(selected_features) // n_cols)  # ceiling division
        fig2, axes = plt.subplots(n_rows_fig, n_cols, figsize=(12, 3.5 * n_rows_fig))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, feat in enumerate(selected_features):
            axes[i].hist(df[feat].dropna(), bins=30, color="#16a34a", edgecolor="white", alpha=0.85)
            axes[i].set_title(feat, fontsize=10)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Count")

        # Hide unused axes
        for j in range(len(selected_features), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Value Distributions", fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Select at least one feature to view its distribution.")

    # ── crop vs feature comparison ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🔎 Feature vs Crop Category</p>', unsafe_allow_html=True)
    st.markdown("Compare the average value of a selected feature across different crop categories.")

    feat_choice = st.selectbox("Select a feature to compare", CROP_FEATURE_COLS)
    crop_filter = st.multiselect(
        "Filter by crops (leave empty to show all)",
        sorted(df["label"].unique()),
        default=[],
    )

    compare_df = df if not crop_filter else df[df["label"].isin(crop_filter)]
    mean_by_crop = compare_df.groupby("label")[feat_choice].mean().sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(mean_by_crop) * 0.4)))
    bars3 = ax3.barh(mean_by_crop.index, mean_by_crop.values, color="#15803d", edgecolor="white")
    ax3.set_xlabel(f"Mean {feat_choice}")
    ax3.set_title(f"Average {feat_choice} by Crop Category")
    ax3.bar_label(bars3, fmt="%.1f", padding=3, fontsize=8)
    ax3.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
