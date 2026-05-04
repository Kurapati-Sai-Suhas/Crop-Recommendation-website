# Data Directory

## Dataset: Crop Recommendation Dataset

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
Based on: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
