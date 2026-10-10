"""
app.py — AgriSmart AI main entry point.

Run with:  streamlit run app.py
"""

import sys
import os

# Make src/ importable when running from the agri-smart-ai/ directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── page config (must be first Streamlit call) ──────────────────────────────────
st.set_page_config(
    page_title="AgriSmart AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Light background, green accent palette */
    .main { background-color: #f9fafb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0 0 0.2rem 0; font-size: 1.4rem; color: #16a34a; }
    .metric-card p  { margin: 0; font-size: 0.85rem; color: #6b7280; }

    /* Section headings */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #15803d;
        border-bottom: 2px solid #bbf7d0;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Info / tip boxes */
    .info-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #166534;
    }
    .warn-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #92400e;
    }

    /* Sidebar nav title */
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── sidebar navigation ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌱 AgriSmart AI")
    st.markdown("*Smart Crop Advisory & Sustainable Farming*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home Dashboard",
            "🌾 Crop Recommendation",
            "📊 Explore Agricultural Data",
            "🔬 Machine Learning Lab",
            "♻️ Sustainable Farming Tips",
            "📈 Model Evaluation",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Supporting UN SDG 2: Zero Hunger")

# ── page routing ────────────────────────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    from pages import page_home
    page_home.show()

elif page == "🌾 Crop Recommendation":
    from pages import page_recommend
    page_recommend.show()

elif page == "📊 Explore Agricultural Data":
    from pages import page_explore
    page_explore.show()

elif page == "🔬 Machine Learning Lab":
    from pages import page_ml_lab
    page_ml_lab.show()

elif page == "♻️ Sustainable Farming Tips":
    from pages import page_tips
    page_tips.show()

elif page == "📈 Model Evaluation":
    from pages import page_evaluation
    page_evaluation.show()
