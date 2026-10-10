"""
page_recommend.py — Crop Recommendation page for AgriSmart AI.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def show() -> None:
    st.title("🌾 Crop Recommendation")
    st.markdown(
        "Enter your soil nutrient levels and environmental conditions below. "
        "The trained model will suggest the most suitable crop based on patterns "
        "in real agricultural data."
    )
    st.markdown(
        '<div class="warn-box">⚠️ This is a data-based suggestion, not a guaranteed agricultural prescription. '
        'Always verify recommendations with a local agronomist before making planting decisions.</div>',
        unsafe_allow_html=True,
    )

    # ── load model ───────────────────────────────────────────────────────────────
    try:
        from src.crop_model import get_or_train_models
        with st.spinner("Loading model…"):
            dt_pipe, rf_pipe = get_or_train_models()
        model_ready = True
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        model_ready = False

    if not model_ready:
        st.stop()

    # ── model selector ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">⚙️ Model Settings</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Choose classification model",
        ["Random Forest (with probabilities)", "Decision Tree"],
        help="Random Forest gives confidence scores. Decision Tree is simpler and easier to explain.",
    )
    pipeline = rf_pipe if "Random" in model_choice else dt_pipe

    # ── input form ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🌱 Enter Your Field Conditions</p>', unsafe_allow_html=True)

    with st.form("crop_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Soil Nutrients**")
            N = st.number_input(
                "Nitrogen (N) — kg/ha",
                min_value=0.0, max_value=200.0, value=50.0, step=1.0,
                help="Available nitrogen in the topsoil. Typical range: 0–200 kg/ha.",
            )
            P = st.number_input(
                "Phosphorus (P) — kg/ha",
                min_value=0.0, max_value=200.0, value=40.0, step=1.0,
                help="Available phosphorus. Typical range: 0–200 kg/ha.",
            )
            K = st.number_input(
                "Potassium (K) — kg/ha",
                min_value=0.0, max_value=300.0, value=40.0, step=1.0,
                help="Available potassium. Typical range: 0–300 kg/ha.",
            )
            ph = st.number_input(
                "Soil pH",
                min_value=0.0, max_value=14.0, value=6.5, step=0.1,
                help="Soil acidity / alkalinity. 7.0 is neutral; most crops prefer 6.0–7.5.",
            )

        with col2:
            st.markdown("**Environmental Conditions**")
            temperature = st.number_input(
                "Temperature — °C",
                min_value=0.0, max_value=50.0, value=25.0, step=0.5,
                help="Average daytime temperature in your area.",
            )
            humidity = st.number_input(
                "Relative Humidity — %",
                min_value=0.0, max_value=100.0, value=70.0, step=1.0,
                help="Average relative humidity (0–100%).",
            )
            rainfall = st.number_input(
                "Rainfall — mm",
                min_value=0.0, max_value=500.0, value=100.0, step=5.0,
                help="Expected or historical annual rainfall in millimetres.",
            )

        col_submit, col_reset = st.columns([1, 1])
        with col_submit:
            submitted = st.form_submit_button("🌾 Recommend Crop", use_container_width=True)
        with col_reset:
            reset = st.form_submit_button("↩️ Reset", use_container_width=True)

    if reset:
        st.rerun()

    # ── prediction ───────────────────────────────────────────────────────────────
    if submitted:
        from src.preprocessing import validate_input_row
        from src.crop_model import predict_crop
        from src.farming_tips import get_tips_for_crop

        user_input = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        errors = validate_input_row(user_input)
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Running model…"):
                crop_name, prob_dict = predict_crop(pipeline, user_input)

            st.markdown("---")
            st.markdown('<p class="section-title">✅ Recommendation Result</p>', unsafe_allow_html=True)

            res_col, conf_col = st.columns([1, 1])
            with res_col:
                st.success(f"### Recommended Crop: **{crop_name.title()}**")
                st.markdown(
                    "This crop is the best match for the soil and climate conditions you provided, "
                    "based on patterns learned from the training dataset."
                )

            with conf_col:
                if prob_dict:
                    top5 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.markdown("**Top 5 Crop Probabilities**")
                    for cn, p in top5:
                        bar_pct = int(p * 100)
                        st.markdown(
                            f"`{cn.title():<18}` "
                            f"{'█' * max(1, bar_pct // 5):<20} {bar_pct}%"
                        )
                else:
                    st.info("Probability estimates are not available for this model.")

            # Input summary
            with st.expander("📋 View Your Entered Conditions"):
                summary_df = pd.DataFrame(
                    [user_input],
                    columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
                )
                summary_df.columns = [
                    "N (kg/ha)", "P (kg/ha)", "K (kg/ha)",
                    "Temp (°C)", "Humidity (%)", "pH", "Rainfall (mm)"
                ]
                st.dataframe(summary_df, use_container_width=True)

            # Sustainable farming tips
            st.markdown("---")
            st.markdown(
                '<p class="section-title">♻️ Sustainable Farming Tips for Your Crop</p>',
                unsafe_allow_html=True,
            )
            tips = get_tips_for_crop(crop_name)
            for category, tip_list in tips.items():
                with st.expander(f"📌 {category}"):
                    for tip in tip_list:
                        st.markdown(f"• {tip}")

            st.markdown(
                '<div class="warn-box">These tips are general educational guidance. '
                'For personalized advice, consult a qualified local agronomist.</div>',
                unsafe_allow_html=True,
            )
