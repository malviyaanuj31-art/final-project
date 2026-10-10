"""
page_tips.py — Sustainable Farming Tips page for AgriSmart AI.
"""

from __future__ import annotations

import streamlit as st


def show() -> None:
    st.title("♻️ Sustainable Farming Tips")
    st.markdown(
        "This page provides practical guidance on sustainable agriculture. "
        "Tips are organised by category. Where a crop was entered in the "
        "Crop Recommendation page, this section may show crop-specific advice."
    )

    st.markdown(
        '<div class="warn-box">'
        "These tips are <strong>general educational guidance</strong>. "
        "For personalised recommendations tailored to your local soil, climate and regulations, "
        "always consult a qualified local agronomist or agricultural extension service."
        "</div>",
        unsafe_allow_html=True,
    )

    # Optional crop filter
    st.markdown("---")
    st.markdown('<p class="section-title">🔎 Crop-Specific Tips (Optional)</p>', unsafe_allow_html=True)

    try:
        from src.data_loader import load_crop_data
        df = load_crop_data()
        crop_options = ["(General — all crops)"] + sorted(df["label"].unique().tolist())
    except Exception:
        crop_options = ["(General — all crops)"]

    selected_crop_display = st.selectbox("Show tips for:", crop_options)
    selected_crop = None if selected_crop_display.startswith("(") else selected_crop_display

    # Load and display tips
    from src.farming_tips import get_tips_for_crop
    tips = get_tips_for_crop(selected_crop)

    for category, tip_list in tips.items():
        st.markdown("---")
        st.markdown(f'<p class="section-title">📌 {category}</p>', unsafe_allow_html=True)
        for tip in tip_list:
            st.markdown(
                f'<div class="info-box">• {tip}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("""
### 🌍 Why Sustainable Farming Matters

Sustainable agriculture aims to meet today's food needs without compromising the ability of future
generations to meet their own. Key goals include:

- **Preserving soil health** — healthy soil is the foundation of food production.
- **Protecting water resources** — agriculture accounts for ~70% of global freshwater withdrawals.
- **Reducing emissions** — sustainable practices can significantly cut greenhouse gas emissions from farming.
- **Supporting biodiversity** — diverse farm ecosystems are more resilient to pests and climate shocks.
- **Improving farmer livelihoods** — efficient, sustainable farms are more profitable over the long term.

*Source: FAO — Sustainable Development Goals, Target 2.4*
    """)
