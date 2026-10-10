"""
page_evaluation.py — Model Evaluation page for AgriSmart AI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def show() -> None:
    st.title("📈 Model Evaluation")
    st.markdown(
        "This page shows the real evaluation results for the crop classification models. "
        "All metrics are calculated on the **held-out test set** (20% of data) that the "
        "model never saw during training."
    )

    st.markdown(
        '<div class="info-box">'
        "Using a separate test set prevents <strong>data leakage</strong>: "
        "reporting training accuracy as if it were test accuracy would give "
        "misleadingly high and unreliable numbers."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── load / train model ───────────────────────────────────────────────────────
    try:
        from src.crop_model import get_or_train_models, evaluate_model

        with st.spinner("Loading models and computing evaluation metrics…"):
            dt_pipe, rf_pipe = get_or_train_models()
            dt_eval = evaluate_model(dt_pipe)
            rf_eval = evaluate_model(rf_pipe)
        models_ready = True
    except Exception as exc:
        st.error(f"Evaluation could not be completed: {exc}")
        models_ready = False

    if not models_ready:
        st.stop()

    # ── model selector ───────────────────────────────────────────────────────────
    st.markdown("---")
    model_choice = st.radio(
        "Select model to evaluate",
        ["Random Forest", "Decision Tree"],
        horizontal=True,
    )
    ev = rf_eval if model_choice == "Random Forest" else dt_eval

    # ── split info ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🔀 Train / Test Split</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", ev["train_size"])
    c2.metric("Test Samples",     ev["test_size"])
    c3.metric("Split Ratio",      "80% / 20%")
    st.caption("A fixed random seed (42) is used for reproducibility.")

    # ── core metrics ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Core Metrics</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",          f"{ev['accuracy']*100:.2f}%")
    m2.metric("Precision (wtd)",   f"{ev['precision']*100:.2f}%")
    m3.metric("Recall (wtd)",      f"{ev['recall']*100:.2f}%")
    m4.metric("F1-Score (wtd)",    f"{ev['f1']*100:.2f}%")

    st.markdown("""
| Metric | What it measures |
|---|---|
| **Accuracy** | Fraction of all predictions that were correct |
| **Precision** | Of predicted positives, how many were truly positive |
| **Recall** | Of actual positives, how many did the model catch |
| **F1-Score** | Harmonic mean of precision and recall — useful when classes are unequal |
*Weighted averages account for the number of samples in each class.*
    """)

    # ── confusion matrix ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🗂 Confusion Matrix</p>', unsafe_allow_html=True)
    st.markdown(
        "Each cell shows how many times a crop was predicted as another crop. "
        "The diagonal (top-left to bottom-right) represents correct predictions."
    )

    cm     = ev["confusion_matrix"]
    labels = ev["classes"]
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(6, n * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_choice}")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── classification report ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📋 Full Classification Report</p>', unsafe_allow_html=True)
    st.code(ev["class_report"], language="text")
    st.caption("Per-class precision, recall, F1-score and support (number of test samples).")

    # ── per-class F1 chart ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📉 Per-Class F1-Score</p>', unsafe_allow_html=True)

    from sklearn.metrics import f1_score
    import pandas as pd

    y_test  = ev["y_test"]
    y_pred  = ev["y_pred"]
    classes = ev["classes"]

    f1_per_class = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    f1_df = pd.DataFrame({"Crop": classes, "F1-Score": f1_per_class}).sort_values("F1-Score")

    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(f1_df) * 0.35)))
    colors = ["#16a34a" if v >= 0.9 else "#f59e0b" if v >= 0.7 else "#ef4444" for v in f1_df["F1-Score"]]
    bars = ax2.barh(f1_df["Crop"], f1_df["F1-Score"], color=colors)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("F1-Score")
    ax2.set_title(f"Per-Crop F1-Score — {model_choice}")
    ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax2.axvline(0.9, color="green", linestyle="--", linewidth=1, alpha=0.6, label="0.9 threshold")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
