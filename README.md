# AgriSmart AI

**Smart Crop Advisory and Sustainable Farming Assistant**

AgriSmart AI is an open-source web application that helps farmers and agricultural planners make
smarter decisions using machine learning. Enter your soil and environmental data to receive
data-driven crop recommendations, explore agricultural datasets, and learn how different ML
techniques work — all in a clean, beginner-friendly interface.

---

## Main Features

| Feature | Description |
|---|---|
| 🌾 **Crop Recommendation** | Predict the most suitable crop from N, P, K, pH, temperature, humidity, and rainfall |
| 📊 **Agricultural Data Explorer** | Browse and visualise the underlying dataset |
| 🔬 **Machine Learning Lab** | Supervised learning, classification, regression, clustering, deep learning and Q-learning demonstrations |
| ♻️ **Sustainable Farming Tips** | Curated, practical advice on soil health, irrigation, crop rotation and more |
| 📈 **Model Evaluation** | Real test-set accuracy, precision, recall, F1-score and confusion matrix |

---

## Machine Learning Techniques Demonstrated

- **Supervised Learning** — concept explanation with crop recommendation as a working example
- **Classification** — Decision Tree and Random Forest for crop category prediction
- **Regression** — Linear Regression and Gradient Boosting for crop yield estimation
- **Decision Trees** — interactive visualisation with depth control
- **Unsupervised Learning / Clustering** — K-Means with elbow plot and PCA scatter plot
- **Deep Learning** — small neural network (optional TensorFlow dependency)
- **Reinforcement Learning / Q-Learning** — grid-world agent simulation

---

## Project Structure

```
agri-smart-ai/
│
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── README.md
│
├── data/
│   ├── crop_recommendation.csv       # Crop dataset (2 200 rows, 22 crops)
│   ├── crop_yield_regression.csv     # Regression demo dataset (500 rows)
│   ├── generate_sample_data.py       # Script to regenerate crop dataset
│   └── generate_regression_data.py  # Script to regenerate regression dataset
│
├── models/                 # Saved .joblib model files (created at first run)
│
├── pages/
│   ├── __init__.py
│   ├── page_home.py        # Page 1: Home Dashboard
│   ├── page_recommend.py   # Page 2: Crop Recommendation
│   ├── page_explore.py     # Page 3: Explore Agricultural Data
│   ├── page_ml_lab.py      # Page 4: Machine Learning Lab
│   ├── page_tips.py        # Page 5: Sustainable Farming Tips
│   └── page_evaluation.py  # Page 6: Model Evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading and validation
│   ├── preprocessing.py    # Feature scaling and input validation
│   ├── crop_model.py       # Classification model training, saving, evaluation
│   ├── regression_demo.py  # Regression demonstration
│   ├── clustering_demo.py  # K-Means clustering demonstration
│   ├── deep_learning_demo.py  # Optional neural network demo
│   ├── q_learning_demo.py  # Q-learning grid-world simulation
│   └── farming_tips.py     # Curated sustainable farming tips
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py   # Dataset validation tests
    ├── test_crop_model.py    # Model training, prediction, evaluation tests
    ├── test_preprocessing.py # Input validation and scaler tests
    └── test_q_learning.py    # Q-learning simulation tests
```

---

## Requirements

- **Python 3.9 or higher** (developed on Python 3.12)
- All core dependencies are free and open-source
- TensorFlow is **optional** — the application works without it

---

## Installation

### Windows (PowerShell)

```powershell
# 1. Navigate to the project folder
cd agri-smart-ai

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Install TensorFlow for the Deep Learning tab
pip install tensorflow
```

### macOS / Linux

```bash
cd agri-smart-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: pip install tensorflow
```

---

## Dataset

The application ships with a **synthetic sample dataset** that mirrors the structure of the
publicly available Crop Recommendation Dataset.

**Dataset columns:** N, P, K, temperature, humidity, ph, rainfall, label

For best results, replace the sample data with the real dataset:

1. Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Rename the file to `crop_recommendation.csv`
3. Place it in the `data/` folder
4. Delete any saved models in `models/` so they are retrained on the new data

To regenerate the sample datasets:

```powershell
python data/generate_sample_data.py
python data/generate_regression_data.py
```

---

## Launching the Application

```powershell
# From the agri-smart-ai/ directory with the virtual environment activated:
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

## Retraining the Model

The model is trained automatically on first run and saved to `models/`.
To force a retrain (for example after updating the dataset):

```powershell
# Delete saved models
Remove-Item models/*.joblib -ErrorAction SilentlyContinue
# Restart the application — models will be retrained on startup
streamlit run app.py
```

---

## Running the Tests

```powershell
# From the agri-smart-ai/ directory with the virtual environment activated:
pip install pytest
pytest tests/ -v
```

Expected output: all tests pass with an accuracy assertion of ≥ 70% on the test split.

---

## Known Limitations

- The sample dataset is **synthetic**. For real-world accuracy, replace it with verified field data.
- Crop recommendations are data-driven suggestions. Always consult a local agronomist before planting decisions.
- The Deep Learning tab requires TensorFlow which is an optional, heavier install.
- Q-Learning is a simplified 6×6 grid simulation — it is educational, not a real farm planning tool.
- Regression results are demonstrated on a synthetic dataset. They do not represent real yield predictions.

---

## Future Improvements

- Integrate with real-time weather APIs for automatic environmental input
- Add support for uploading a custom CSV dataset
- Expand the crop database beyond the current 22 categories
- Add multilingual support for non-English speaking farmers
- Provide a mobile-friendly responsive layout
- Include soil map overlays using GIS data

---

## Connection to UN SDG 2: Zero Hunger

AgriSmart AI is aligned with **UN Sustainable Development Goal 2: Zero Hunger**, which calls for
ending hunger, achieving food security, improving nutrition, and promoting sustainable agriculture
by 2030. Better crop planning helps reduce input waste, improve yields, and support long-term
soil health — contributing directly to this goal.

---

*AgriSmart AI — Open-source agricultural intelligence for better farming decisions.*
