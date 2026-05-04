from backend.services import explainer, predictor
import numpy as np

# Input from user
data = {"N":22.01,"P":17.75,"K":41.53,"temperature":21.12,"humidity":90.55,"ph":6.83,"rainfall":105.82}

# Ensure explainer is initialised
explainer._load_explainer()
# Get raw and scaled feature arrays
raw, scaled = predictor.get_feature_array(data)

# Compute SHAP values (raw, without rounding)
shap_vals = explainer._explainer.shap_values(raw)

le = predictor.get_label_encoder()
classes = list(le.classes_) if le is not None else None

print('LABEL_CLASSES:', classes)
print('RAW_FEATURES:', raw.tolist())
print('SCALED_FEATURES:', (scaled.tolist() if scaled is not None else None))

# Print shap values with full precision
if isinstance(shap_vals, list):
    for idx, arr in enumerate(shap_vals):
        print(f'CLASS_INDEX:{idx} SHAP_VALUES:', np.array(arr).astype(float).tolist())
else:
    print('SHAP_ARRAY:', np.array(shap_vals).astype(float).tolist())

# Also print RandomForest predict_proba for transparency
rf = predictor.get_rf_model()
if rf is not None:
    proba = rf.predict_proba(raw if not (scaled is not None) else scaled)[0]
    print('PREDICT_PROBA:', proba.tolist())
    pred_idx = rf.predict(raw if not (scaled is not None) else scaled)[0]
    print('PRED_INDEX:', int(pred_idx))

print('DONE')
