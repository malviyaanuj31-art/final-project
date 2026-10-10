"""
Generate a sample agricultural regression dataset for AgriSmart AI.

Target variable: crop yield (tonnes per hectare)
Features: rainfall, temperature, fertilizer_kg_per_ha, soil_nitrogen, irrigation_count

NOTE: This is a synthetic demonstration dataset.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
N = 500


def generate_regression_dataset() -> pd.DataFrame:
    rainfall        = rng.uniform(50,  300, N)
    temperature     = rng.uniform(15,  35,  N)
    fertilizer      = rng.uniform(50,  250, N)
    soil_nitrogen   = rng.uniform(10,  120, N)
    irrigation      = rng.integers(0,  8,   N).astype(float)

    # Synthetic yield formula with some noise
    yield_tph = (
        0.015 * rainfall
        + 0.12  * fertilizer
        + 0.08  * soil_nitrogen
        + 0.4   * irrigation
        - 0.05  * (temperature - 25) ** 2
        + rng.normal(0, 1.2, N)
    )
    yield_tph = np.clip(yield_tph, 0.5, 30).round(2)

    df = pd.DataFrame({
        "rainfall_mm":          rainfall.round(1),
        "temperature_C":        temperature.round(1),
        "fertilizer_kg_per_ha": fertilizer.round(1),
        "soil_nitrogen":        soil_nitrogen.round(1),
        "irrigation_count":     irrigation.astype(int),
        "yield_tph":            yield_tph,
    })
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "crop_yield_regression.csv")
    df = generate_regression_dataset()
    df.to_csv(out_path, index=False)
    print(f"Regression dataset saved to {out_path}  ({len(df)} rows)")
