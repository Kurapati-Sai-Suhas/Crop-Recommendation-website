# Data Directory

## ⚠️ This dataset is synthetic

`crop_data.csv` is **generated**, not observed. `generate_data.py` draws each
feature independently from a hardcoded per-crop Gaussian (`CROP_PROFILES`),
using distribution parameters loosely modelled on the Kaggle dataset linked
below.

Two consequences worth understanding before trusting any metric computed here:

1. **Features are independent by construction.** Real soil chemistry is
   correlated — nitrogen, pH and rainfall move together. Here they do not.
2. **Gaussian Naive Bayes is the Bayes-optimal classifier for this data**,
   because the generator *is* a Gaussian Naive Bayes model. Its ~99% accuracy
   measures the generator, not agronomy. Every other capable model lands
   within ~0.7 percentage points for the same reason.

The generator is useful as a reproducible fixture for tests and demos. It is
not a substitute for the real dataset. To get meaningful numbers, download the
[Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
(2,200 rows, identical schema) and retrain.

---

## Dataset: Crop Recommendation Dataset (synthetic)

### Overview
- **Rows**: 2200 (100 samples per crop)
- **Features**: 7 input features + 1 label
- **Crops**: 22 types

### Features

| Feature | Unit | Description |
|---------|------|-------------|
| N | ratio | Nitrogen content in soil |
| P | ratio | Phosphorus content in soil |
| K | ratio | Potassium content in soil |
| temperature | °C | Average temperature |
| humidity | % | Relative humidity |
| ph | 0–14 | pH value of soil |
| rainfall | mm | Annual rainfall |
| label | — | Crop name (target variable) |

### Crops (22 classes)
rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean,
blackgram, lentil, pomegranate, banana, mango, grapes, watermelon,
muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

### Generating the Dataset
```bash
python data/generate_data.py
```

### Source
Distribution parameters were hand-written in `generate_data.py`, loosely based
on: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
**No rows from that dataset are present here.**
