"""
page_ml_lab.py — Machine Learning Lab for AgriSmart AI.

Covers: Supervised Learning, Classification (Decision Tree),
        Regression, Clustering (K-Means), Deep Learning, Q-Learning.
"""

from __future__ import annotations

import streamlit as st


def show() -> None:
    st.title("🔬 Machine Learning Lab")
    st.markdown(
        "This page demonstrates the main machine learning techniques used in AgriSmart AI. "
        "Each section includes a brief explanation and a live demonstration."
    )

    tab_labels = [
        "A · Supervised Learning",
        "B · Classification (Decision Tree)",
        "C · Regression",
        "D · Decision Tree Visualizer",
        "E · Clustering (K-Means)",
        "F · Deep Learning",
        "G · Q-Learning",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _tab_supervised()
    with tabs[1]:
        _tab_classification()
    with tabs[2]:
        _tab_regression()
    with tabs[3]:
        _tab_dt_visualizer()
    with tabs[4]:
        _tab_clustering()
    with tabs[5]:
        _tab_deep_learning()
    with tabs[6]:
        _tab_q_learning()


# ── A: Supervised Learning ────────────────────────────────────────────────────

def _tab_supervised() -> None:
    st.subheader("A · Supervised Learning")
    st.markdown("""
**Supervised learning** is a type of machine learning where a model is trained on *labelled* examples —
data where the correct answer is already known.

In AgriSmart AI, the training data includes soil readings and weather measurements paired with the
known crop that was successfully grown under those conditions. The model learns the patterns that
link inputs (N, P, K, temperature, humidity, pH, rainfall) to outputs (crop label).

Once trained, the model can predict the most suitable crop for *new*, unseen field conditions.

### Key concepts

| Concept | Explanation |
|---|---|
| **Features (X)** | The inputs: N, P, K, temperature, humidity, pH, rainfall |
| **Label / Target (y)** | The answer to learn: the crop name |
| **Training data** | 80% of the dataset, used to fit the model |
| **Test data** | 20% held back to measure real-world accuracy |
| **Prediction** | The model's output for new input data |

### Why labelled data matters
Without labels, the model has no way to check whether it is making correct predictions.
Labelled data acts as the teacher — the model adjusts itself to minimise mistakes on the
training examples and (ideally) generalises well to new ones.
    """)

    st.markdown(
        '<div class="info-box">👉 The <strong>Crop Recommendation</strong> feature uses supervised learning. '
        "You provide the features; the model predicts the label.</div>",
        unsafe_allow_html=True,
    )


# ── B: Classification ──────────────────────────────────────────────────────────

def _tab_classification() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    st.subheader("B · Classification")
    st.markdown("""
**Classification** is supervised learning where the output is a *category* (not a number).
In crop recommendation, the model predicts which crop category a set of field conditions belongs to.

AgriSmart AI trains two classifiers:
- **Decision Tree** — makes sequential yes/no splits on features. Easy to visualise and explain.
- **Random Forest** — builds many decision trees and combines their votes. Usually more accurate.
    """)

    try:
        from src.crop_model import get_or_train_models, evaluate_model
        from src.data_loader import load_crop_data

        with st.spinner("Training / loading classifiers…"):
            dt_pipe, rf_pipe = get_or_train_models()
            dt_eval = evaluate_model(dt_pipe)
            rf_eval = evaluate_model(rf_pipe)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Decision Tree**")
            st.metric("Test Accuracy", f"{dt_eval['accuracy']*100:.2f}%")
            st.metric("F1-Score (weighted)", f"{dt_eval['f1']*100:.2f}%")
        with col2:
            st.markdown("**Random Forest**")
            st.metric("Test Accuracy", f"{rf_eval['accuracy']*100:.2f}%")
            st.metric("F1-Score (weighted)", f"{rf_eval['f1']*100:.2f}%")

        st.info(
            "These metrics are calculated on the **held-out test set** (20% of data). "
            "The model never saw these examples during training."
        )

    except Exception as exc:
        st.error(f"Could not load models: {exc}")


# ── C: Regression ─────────────────────────────────────────────────────────────

def _tab_regression() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    st.subheader("C · Regression")
    st.markdown("""
**Regression** is supervised learning where the output is a *continuous number* rather than a category.

> **Note:** Crop *recommendation* is a classification task (which crop?).
> This regression demonstration uses a **separate synthetic demonstration dataset**
> to predict **crop yield in tonnes per hectare** from environmental and management inputs.
> It is clearly labelled as sample data and is provided for educational purposes only.

**Dataset features:**
| Feature | Description |
|---|---|
| rainfall_mm | Annual rainfall (mm) |
| temperature_C | Average temperature (°C) |
| fertilizer_kg_per_ha | Fertilizer applied (kg/ha) |
| soil_nitrogen | Soil nitrogen level (kg/ha) |
| irrigation_count | Number of irrigation cycles |

**Target:** `yield_tph` — crop yield in tonnes per hectare
    """)

    try:
        from src.regression_demo import train_regression_models

        with st.spinner("Training regression models…"):
            res = train_regression_models()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Linear Regression**")
            for k, v in res["lr_metrics"].items():
                st.metric(k, v)
        with col2:
            st.markdown("**Gradient Boosting Regressor**")
            for k, v in res["gb_metrics"].items():
                st.metric(k, v)

        st.markdown("""
**Metric explanations:**
- **MAE** (Mean Absolute Error): average absolute difference between predicted and actual yield.
- **RMSE** (Root Mean Squared Error): similar to MAE but penalises large errors more.
- **R²** (R-squared): how much of the variance in yield is explained by the model (1.0 = perfect).
        """)

        # Actual vs predicted scatter
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, preds, title in zip(
            axes,
            [res["lr_pred"], res["gb_pred"]],
            ["Linear Regression", "Gradient Boosting"],
        ):
            ax.scatter(res["y_test"], preds, alpha=0.4, s=20, color="#16a34a")
            lo = min(res["y_test"].min(), preds.min())
            hi = max(res["y_test"].max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
            ax.set_xlabel("Actual Yield (t/ha)")
            ax.set_ylabel("Predicted Yield (t/ha)")
            ax.set_title(f"{title}\nActual vs Predicted")
            ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Feature importance from GB
        st.markdown("**Feature Importances (Gradient Boosting)**")
        imp = res["importances"]
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        sorted_imp = sorted(imp.items(), key=lambda x: x[1])
        ax2.barh([k for k, _ in sorted_imp], [v for _, v in sorted_imp], color="#15803d")
        ax2.set_xlabel("Importance")
        ax2.set_title("Which features most influence predicted yield?")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as exc:
        st.error(f"Regression demo error: {exc}")


# ── D: Decision Tree Visualizer ───────────────────────────────────────────────

def _tab_dt_visualizer() -> None:
    import matplotlib.pyplot as plt
    from sklearn import tree as sk_tree

    st.subheader("D · Decision Tree Visualizer")
    st.markdown("""
A **Decision Tree** learns a series of if-then rules by splitting data at each step based on
the feature that best separates the crop categories. The rules can be read from top to bottom.

- **Root node** — the first split (most important question).
- **Branches** — the yes/no paths based on a condition.
- **Leaf nodes** — the final predicted crop category.
    """)

    try:
        from src.crop_model import get_or_train_models
        from src.data_loader import CROP_FEATURE_COLS

        with st.spinner("Loading Decision Tree…"):
            dt_pipe, _ = get_or_train_models()

        dt_model = dt_pipe.named_steps["classifier"]

        max_depth_viz = st.slider(
            "Tree depth to visualize (smaller = more readable)",
            min_value=2, max_value=6, value=3,
        )

        fig, ax = plt.subplots(figsize=(max_depth_viz * 4, max_depth_viz * 2.5))
        sk_tree.plot_tree(
            dt_model,
            feature_names=CROP_FEATURE_COLS,
            class_names=dt_pipe.classes_,
            filled=True,
            rounded=True,
            max_depth=max_depth_viz,
            fontsize=7,
            ax=ax,
        )
        ax.set_title(f"Decision Tree (showing top {max_depth_viz} levels)", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"The full tree has depth {dt_model.get_depth()} and "
            f"{dt_model.get_n_leaves()} leaf nodes. "
            "Only a readable portion is shown above."
        )

    except Exception as exc:
        st.error(f"Decision tree visualization error: {exc}")


# ── E: Clustering ─────────────────────────────────────────────────────────────

def _tab_clustering() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    st.subheader("E · Unsupervised Learning — K-Means Clustering")
    st.markdown("""
**Unsupervised learning** finds patterns in data *without* using labels. The model is given
features only — no crop names — and must discover structure on its own.

**K-Means clustering** groups data points into *k* clusters based on feature similarity.
Points that are closer together in the feature space are assigned to the same cluster.

> **Important:** Clusters represent natural groupings in the data. They do **not** automatically
> correspond to expert-defined crop categories or agronomic recommendations.
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of clusters (k)", 2, 15, 5)
    with col2:
        st.markdown("")
        run_btn = st.button("▶ Run Clustering", use_container_width=True)

    if run_btn:
        try:
            from src.clustering_demo import run_clustering, compute_elbow

            with st.spinner("Running K-Means…"):
                result = run_clustering(n_clusters)
                elbow  = compute_elbow(max_k=10)

            res_df = result["result_df"]

            # Scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            # Left: cluster scatter
            ax1 = axes[0]
            colors = cm.get_cmap("tab20", n_clusters)
            for c in range(n_clusters):
                mask = res_df["cluster"] == c
                ax1.scatter(
                    res_df.loc[mask, "pca_x"],
                    res_df.loc[mask, "pca_y"],
                    s=15, alpha=0.6,
                    color=colors(c),
                    label=f"Cluster {c}",
                )
            ax1.set_xlabel(f"PCA Component 1 ({result['pca_variance_explained'][0]*100:.1f}% variance)")
            ax1.set_ylabel(f"PCA Component 2 ({result['pca_variance_explained'][1]*100:.1f}% variance)")
            ax1.set_title(f"K-Means Clusters (k={n_clusters})\n2D PCA projection")
            ax1.legend(fontsize=7, ncol=2, loc="best")

            # Right: elbow plot
            ax2 = axes[1]
            ax2.plot(elbow["k_values"], elbow["inertia_values"], "o-", color="#16a34a", linewidth=2)
            ax2.axvline(x=n_clusters, color="red", linestyle="--", alpha=0.7, label=f"Selected k={n_clusters}")
            ax2.set_xlabel("Number of Clusters (k)")
            ax2.set_ylabel("Inertia (within-cluster sum of squares)")
            ax2.set_title("Elbow Plot — Choosing k")
            ax2.legend(fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Inertia (selected k)", result["inertia"])

            with st.expander("📋 Cluster Centre Feature Values"):
                st.dataframe(result["centres_df"].round(2), use_container_width=True)

        except Exception as exc:
            st.error(f"Clustering error: {exc}")

    st.markdown("""
**Classification vs Clustering:**

| | Classification | Clustering |
|---|---|---|
| Uses labels? | ✅ Yes | ❌ No |
| Output | Known category | Discovered group |
| Example | "This is rice" | "Fields A/C/F are similar" |
| Type | Supervised | Unsupervised |
    """)


# ── F: Deep Learning ──────────────────────────────────────────────────────────

def _tab_deep_learning() -> None:
    import matplotlib.pyplot as plt

    st.subheader("F · Deep Learning Fundamentals")
    st.markdown("""
**Deep learning** uses artificial neural networks with multiple layers to learn complex patterns.
This demonstration trains a small neural network on the crop classification dataset.

### Network architecture
- **Input layer**: 7 nodes — one for each feature (N, P, K, temperature, humidity, pH, rainfall)
- **Hidden layer 1**: 64 neurons with ReLU activation
- **Dropout**: randomly disables 20% of neurons during training to prevent overfitting
- **Hidden layer 2**: 32 neurons with ReLU activation
- **Output layer**: one node per crop class, Softmax activation (outputs probabilities)

### Key concepts
| Concept | Meaning |
|---|---|
| **Epoch** | One complete pass through the training data |
| **Loss** | How wrong the model currently is (lower = better) |
| **Accuracy** | Fraction of correct predictions |
| **Dropout** | Technique to prevent memorising training data |
| **Softmax** | Converts output to probabilities that sum to 1 |
    """)

    try:
        from src.deep_learning_demo import is_tf_available

        if not is_tf_available():
            st.warning(
                "**TensorFlow is not installed.** To enable this feature:\n\n"
                "```\npip install tensorflow\n```\n\n"
                "The rest of the application works without it."
            )
            st.stop()

        epochs = st.slider("Training epochs", 5, 50, 20)
        run_dl = st.button("▶ Train Neural Network", use_container_width=True)

        if run_dl:
            from src.deep_learning_demo import train_neural_network

            with st.spinner(f"Training neural network for {epochs} epochs…"):
                result = train_neural_network(epochs=epochs)

            col1, col2 = st.columns(2)
            col1.metric("Test Accuracy", f"{result['test_accuracy']*100:.2f}%")
            col2.metric("Test Loss", f"{result['test_loss']:.4f}")

            # Training curves
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(result["history_train_acc"], label="Train", color="#16a34a")
            axes[0].plot(result["history_val_acc"],   label="Validation", color="#f59e0b", linestyle="--")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Accuracy Over Training Epochs")
            axes[0].legend()

            axes[1].plot(result["history_train_loss"], label="Train", color="#16a34a")
            axes[1].plot(result["history_val_loss"],   label="Validation", color="#f59e0b", linestyle="--")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Loss Over Training Epochs")
            axes[1].legend()

            plt.suptitle("Neural Network Training Progress", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Model Architecture Summary"):
                st.code(result["model_summary"], language="text")

    except Exception as exc:
        st.error(f"Deep learning demo error: {exc}")


# ── G: Q-Learning ─────────────────────────────────────────────────────────────

def _tab_q_learning() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    st.subheader("G · Reinforcement Learning — Q-Learning")
    st.markdown("""
**Reinforcement learning** is a type of machine learning where an *agent* learns by trial and error,
receiving rewards for good actions and penalties for bad ones.

**This is an educational grid-world simulation.** A virtual agent learns to navigate a 6×6 grid
from the Start cell to the Goal cell while avoiding obstacles.

> ⚠️ This simulation does not control real farming equipment or make validated farm-management decisions.

### Q-Learning concepts
| Concept | Explanation |
|---|---|
| **State** | The agent's current position in the grid |
| **Action** | Move up, down, left, or right |
| **Reward** | +10 for reaching the goal; −1 for hitting a wall; −0.05 per step |
| **Q-table** | Stores the learned value of each (state, action) pair |
| **Learning rate (α)** | How much the agent updates its Q-values each step |
| **Discount factor (γ)** | How much the agent values future rewards vs immediate ones |
| **Exploration (ε)** | Probability of trying a random action instead of the best known one |
    """)

    # Parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha    = st.slider("Learning rate (α)", 0.01, 1.0, 0.1, 0.01)
    with col2:
        gamma    = st.slider("Discount factor (γ)", 0.5, 0.99, 0.9, 0.01)
    with col3:
        epsilon  = st.slider("Exploration rate (ε)", 0.0, 1.0, 0.2, 0.05)
    with col4:
        episodes = st.slider("Training episodes", 100, 2000, 500, 100)

    run_ql = st.button("▶ Train Q-Learning Agent", use_container_width=True)

    if run_ql:
        try:
            from src.q_learning_demo import run_q_learning

            with st.spinner(f"Training for {episodes} episodes…"):
                result = run_q_learning(
                    alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes
                )

            fig, axes = plt.subplots(1, 2, figsize=(13, 6))

            # Left: grid + path
            ax1 = axes[0]
            grid = result["grid"]
            rows, cols = result["rows"], result["cols"]
            best_path  = result["best_path"]
            path_set   = set(best_path)

            for r in range(rows):
                for c in range(cols):
                    cell = (r, c)
                    if grid[r, c] == 1:
                        color = "#374151"  # obstacle
                    elif cell == result["goal"]:
                        color = "#16a34a"  # goal
                    elif cell == result["start"]:
                        color = "#2563eb"  # start
                    elif cell in path_set:
                        color = "#fde68a"  # path
                    else:
                        color = "#f3f4f6"  # free
                    rect = plt.Rectangle((c, rows - r - 1), 1, 1, color=color, ec="white", lw=1)
                    ax1.add_patch(rect)

                    if cell == result["start"]:
                        ax1.text(c + 0.5, rows - r - 0.5, "S", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
                    elif cell == result["goal"]:
                        ax1.text(c + 0.5, rows - r - 0.5, "G", ha="center", va="center", fontsize=11, color="white", fontweight="bold")
                    elif cell in path_set and cell not in (result["start"], result["goal"]):
                        ax1.text(c + 0.5, rows - r - 0.5, "·", ha="center", va="center", fontsize=14, color="#92400e")

            ax1.set_xlim(0, cols)
            ax1.set_ylim(0, rows)
            ax1.set_aspect("equal")
            ax1.axis("off")
            ax1.set_title(f"Learned Path (length={len(best_path)})", fontsize=11)

            legend_elements = [
                mpatches.Patch(color="#2563eb", label="Start (S)"),
                mpatches.Patch(color="#16a34a", label="Goal (G)"),
                mpatches.Patch(color="#fde68a", label="Learned path"),
                mpatches.Patch(color="#374151", label="Obstacle"),
            ]
            ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

            # Right: reward over time (smoothed)
            ax2 = axes[1]
            rewards = result["reward_history"]
            window  = max(1, episodes // 50)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax2.plot(rewards, color="#d1fae5", linewidth=0.5, alpha=0.6, label="Episode reward")
            ax2.plot(
                range(window - 1, len(rewards)),
                smoothed, color="#16a34a", linewidth=2, label=f"Smoothed (window={window})"
            )
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Total Reward")
            ax2.set_title("Reward Over Training Episodes")
            ax2.legend(fontsize=8)

            plt.suptitle("Q-Learning Agent Results", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            reached_goal = result["best_path"][-1] == result["goal"] if result["best_path"] else False
            if reached_goal:
                st.success(f"✅ Agent successfully learned to reach the goal in {len(best_path)} steps!")
            else:
                st.warning("The agent did not find a complete path to the goal with these parameters. Try more episodes.")

        except Exception as exc:
            st.error(f"Q-learning demo error: {exc}")
