"""
page_home.py — Home Dashboard for AgriSmart AI.
"""

from __future__ import annotations

import streamlit as st


def show() -> None:
    st.title("🌱 Welcome to AgriSmart AI")
    st.markdown(
        "**AgriSmart AI** helps farmers and agricultural planners make smarter decisions "
        "by combining soil data, environmental conditions and machine learning. "
        "Enter your field conditions to receive data-driven crop recommendations and "
        "sustainable farming guidance."
    )

    st.markdown("---")

    # ── metrics row ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 At a Glance</p>', unsafe_allow_html=True)

    try:
        from src.data_loader import load_crop_data
        from src.crop_model import get_or_train_models, evaluate_model

        with st.spinner("Loading data and model…"):
            df = load_crop_data()
            _dt_pipe, rf_pipe = get_or_train_models()
            eval_res = evaluate_model(rf_pipe)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset Size", f"{len(df):,} rows")
        with col2:
            st.metric("Crop Categories", df["label"].nunique())
        with col3:
            st.metric("Input Features", "7")
        with col4:
            st.metric("Model Accuracy (test)", f"{eval_res['accuracy']*100:.1f}%")

    except Exception as exc:
        st.warning(f"Metrics unavailable — {exc}")

    st.markdown("---")

    # ── features overview ────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🛠 What AgriSmart AI Offers</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**🌾 Crop Recommendation**
Enter your soil and weather data to receive a data-based crop suggestion with confidence information.

**📊 Agricultural Data Explorer**
Browse and visualize the underlying crop dataset — distribution charts, summary statistics, and feature comparisons.

**♻️ Sustainable Farming Tips**
Practical, curated advice on soil health, water conservation, crop rotation, and responsible fertilizer use.
""")
    with col_b:
        st.markdown("""
**🔬 Machine Learning Lab**
Explore supervised learning, classification, regression, clustering, deep learning, and Q-learning with clear explanations.

**📈 Model Evaluation**
View real accuracy, precision, recall, F1-score, and confusion matrix results calculated on held-out test data.
""")

    st.markdown("---")

    # ── workflow ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">⚙️ How It Works</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1 — Enter Conditions**\n\nProvide your soil nutrient levels, pH, temperature, humidity, and rainfall.")
    with c2:
        st.info("**Step 2 — Analyse Data**\n\nThe trained model processes your inputs and identifies the most suitable crop pattern.")
    with c3:
        st.info("**Step 3 — Get Recommendations**\n\nReceive your top crop suggestion, confidence score, and relevant sustainable farming tips.")

    st.markdown("---")

    # ── SDG 2 section ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🌍 Supporting SDG 2: Zero Hunger</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
<strong>UN Sustainable Development Goal 2</strong> aims to end hunger, achieve food security, improve 
nutrition, and promote sustainable agriculture by 2030.
<br><br>
AgriSmart AI contributes to this goal by helping farmers:
<ul>
  <li>Choose crops that are well-matched to their local soil and climate conditions.</li>
  <li>Reduce input waste through data-driven nutrient and water management guidance.</li>
  <li>Adopt sustainable practices that preserve soil health for future harvests.</li>
  <li>Understand how machine learning can support better agricultural planning.</li>
</ul>
<em>Better decisions today mean more food, less waste, and healthier land for tomorrow.</em>
</div>
""", unsafe_allow_html=True)

    # ── call to action ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.success("👈 Use the sidebar to navigate. Start with **🌾 Crop Recommendation** to try your first prediction.")
