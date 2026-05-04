# -*- coding: utf-8 -*-
"""
============================================================
data/generate_data.py
============================================================
Generates a realistic synthetic dataset based on the well-known
Kaggle Crop Recommendation dataset distributions.
22 crops | 2200 rows | 7 features + 1 label

Run once: python data/generate_data.py
============================================================
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Each tuple: (mean_N, std_N, mean_P, std_P, mean_K, std_K,
#              mean_temp, std_temp, mean_humid, std_humid,
#              mean_ph, std_ph, mean_rain, std_rain)
CROP_PROFILES = {
    "rice":         (80, 6,  40, 5,  40, 5,  23, 1,  82, 3,  6.0, 0.3, 200, 20),
    "maize":        (78, 6,  48, 5,  20, 4,  22, 2,  65, 5,  6.5, 0.3, 60,  10),
    "chickpea":     (40, 5,  67, 5,  79, 5,  18, 2,  16, 3,  7.0, 0.3, 80,  10),
    "kidneybeans":  (20, 4,  67, 5,  20, 4,  19, 2,  21, 4,  5.7, 0.3, 65,  10),
    "pigeonpeas":   (20, 4,  67, 5,  20, 4,  28, 2,  48, 5,  5.7, 0.3, 150, 15),
    "mothbeans":    (20, 4,  45, 5,  45, 5,  28, 2,  53, 5,  3.5, 0.3, 50,  8),
    "mungbean":     (20, 4,  47, 5,  20, 4,  28, 2,  86, 3,  6.6, 0.3, 48,  8),
    "blackgram":    (40, 5,  67, 5,  19, 4,  30, 2,  66, 5,  7.0, 0.3, 68,  10),
    "lentil":       (18, 3,  68, 5,  19, 3,  24, 2,  64, 5,  6.9, 0.3, 45,  8),
    "pomegranate":  (18, 3,  18, 3,  40, 5,  22, 2,  90, 3,  6.7, 0.3, 107, 12),
    "banana":       (100,8,  82, 6,  50, 6,  27, 2,  80, 4,  6.0, 0.3, 105, 12),
    "mango":        (20, 4,  27, 4,  30, 4,  31, 2,  50, 5,  6.0, 0.3, 95,  10),
    "grapes":       (23, 4,  133,8,  200,12, 24, 2,  81, 3,  6.0, 0.3, 69,  10),
    "watermelon":   (99, 7,  17, 3,  50, 5,  25, 2,  85, 3,  6.5, 0.3, 50,  8),
    "muskmelon":    (100,7,  17, 3,  50, 5,  28, 2,  92, 2,  6.3, 0.3, 25,  5),
    "apple":        (20, 4,  134,8,  199,12, 22, 2,  92, 2,  5.8, 0.3, 115, 12),
    "orange":       (19, 3,  16, 3,  10, 3,  22, 2,  92, 2,  7.0, 0.3, 110, 12),
    "papaya":       (49, 5,  59, 5,  50, 5,  34, 2,  92, 2,  6.7, 0.3, 146, 15),
    "coconut":      (21, 4,  16, 3,  30, 4,  27, 2,  94, 2,  5.9, 0.3, 175, 18),
    "cotton":       (117,9,  46, 5,  19, 4,  24, 2,  80, 4,  6.9, 0.3, 80,  10),
    "jute":         (78, 6,  46, 5,  40, 5,  25, 2,  80, 4,  6.7, 0.3, 175, 18),
    "coffee":       (101,7,  28, 4,  29, 4,  25, 2,  58, 5,  6.7, 0.3, 155, 15),
}

SAMPLES_PER_CROP = 100  # 22 x 100 = 2200 rows


def generate_crop_data():
    """Generate synthetic crop recommendation data."""
    rows = []
    for crop, params in CROP_PROFILES.items():
        (mn, sn, mp, sp, mk, sk,
         mt, st, mh, sh, mph, sph, mr, sr) = params

        for _ in range(SAMPLES_PER_CROP):
            rows.append({
                "N":           max(0, round(np.random.normal(mn, sn), 2)),
                "P":           max(0, round(np.random.normal(mp, sp), 2)),
                "K":           max(0, round(np.random.normal(mk, sk), 2)),
                "temperature": round(np.random.normal(mt, st), 2),
                "humidity":    round(np.clip(np.random.normal(mh, sh), 1, 100), 2),
                "ph":          round(np.clip(np.random.normal(mph, sph), 3.0, 9.0), 2),
                "rainfall":    max(0, round(np.random.normal(mr, sr), 2)),
                "label":       crop,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "crop_data.csv")
    df = generate_crop_data()
    df.to_csv(out_path, index=False)
    print(f"[OK] Dataset generated: {len(df)} rows -> {out_path}")
    print(f"   Crops: {sorted(df['label'].unique())}")
    print(f"   Samples per crop: {df['label'].value_counts().to_dict()}")
