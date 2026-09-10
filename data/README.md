# Data Directory

## Provenance

`crop_data.csv` is the public **Crop Recommendation Dataset** (Atharva Ingle,
Kaggle). Kaggle requires authentication, so `download_data.py` fetches a
byte-identical public mirror and verifies it before writing anything to disk:

- **MD5** must equal `fa83739710074b21a20b156034538279`
- **Structure** must be 2,200 rows × 8 columns across 22 crops

Both checks run every time. A changed mirror stops the download rather than
silently retraining the models on different data.

```bash
python data/download_data.py            # fetch + verify
python data/download_data.py --force    # re-download
```

## Schema

| Feature | Unit | Description |
|---------|------|-------------|
| N | ratio | Nitrogen content in soil |
| P | ratio | Phosphorus content in soil |
| K | ratio | Potassium content in soil |
| temperature | °C | Average temperature |
| humidity | % | Relative humidity |
| ph | 0–14 | Soil pH |
| rainfall | mm | Annual rainfall |
| label | — | Crop name (target) |

2,200 rows, 22 crops, **exactly 100 samples per crop**. No missing values and
no duplicate rows — verified in [`../eda/EDA_REPORT.md`](../eda/EDA_REPORT.md),
which recomputes all of this from the file itself.

## Crops (22 classes)

rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean,
blackgram, lentil, pomegranate, banana, mango, grapes, watermelon,
muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

## ⚠️ Read the accuracy figures carefully

This is a teaching benchmark, and it behaves like one.

**The task is nearly solved before you start.** An untuned linear discriminant
reaches 0.967 in 5-fold CV and 1-NN reaches 0.974, against 0.9955 for a
grid-searched Random Forest. The spread between the leading models is smaller
than the fold-to-fold standard deviation, so this data cannot support a claim
that one model is better than another.

**It is probably not measured data.** Each crop's feature values are cleanly
bounded with no tails: across all seven features there is not a single
within-class IQR outlier, where roughly 15 would be expected from normally
distributed measurements. Within-class feature correlations also sit at the
level you would expect from sampling noise alone.

That is fine for demonstrating a modelling workflow, which is what this
repository is for. It is not a basis for agronomic advice.

## Previous versions

Before this, the file was produced by a `generate_data.py` script that drew
each feature from a hand-written per-crop Gaussian. That generator drew every
feature independently — precisely the generative model `GaussianNB` assumes —
so any accuracy measured on it scored the generator rather than agronomy. It
was replaced by the verified download above.
