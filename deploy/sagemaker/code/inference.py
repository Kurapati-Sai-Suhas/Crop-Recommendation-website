# -*- coding: utf-8 -*-
"""
============================================================
deploy/sagemaker/code/inference.py
============================================================
SageMaker inference handlers for the crop recommendation model.

SageMaker's SKLearn container calls four functions, in this order,
for every request:

    model_fn(model_dir)          once, at container start
    input_fn(body, content_type) deserialise the request
    predict_fn(data, model)      run the model
    output_fn(pred, accept)      serialise the response

The container provides defaults for all but model_fn. They are all
overridden here because the defaults assume a bare estimator taking a
numpy array, and this model is three artifacts that must be applied in
a fixed order -- scaler, then forest, then label encoder. Letting the
default input_fn hand a raw array straight to the model would skip
scaling entirely and return confident nonsense.

Feature order is the other thing the defaults cannot know. The model
was fitted on columns in FEATURES order; a JSON object has no inherent
order, so input_fn rebuilds the vector explicitly rather than trusting
dict iteration.
============================================================
"""

import json
import os

import joblib
import numpy as np

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Same agronomic bounds the FastAPI service enforces via Pydantic. Repeated
# here because a SageMaker endpoint can be called directly, bypassing the API,
# and an endpoint that accepts anything will happily score nonsense.
BOUNDS = {
    "N": (0, 200), "P": (0, 200), "K": (0, 210),
    "temperature": (0, 50), "humidity": (0, 100),
    "ph": (0, 14), "rainfall": (0, 500),
}


def model_fn(model_dir):
    """
    Load the artifacts once, at container start.

    Returns a dict rather than a bare estimator: inference needs the
    scaler and the label encoder too, and keeping them together makes it
    impossible to load a model without the transform that belongs to it.
    """
    bundle = {
        "model":   joblib.load(os.path.join(model_dir, "RandomForest.joblib")),
        "scaler":  joblib.load(os.path.join(model_dir, "scaler.joblib")),
        "encoder": joblib.load(os.path.join(model_dir, "label_encoder.joblib")),
    }

    if bundle["scaler"].n_features_in_ != len(FEATURES):
        raise ValueError(
            f"Scaler expects {bundle['scaler'].n_features_in_} features, "
            f"but this handler sends {len(FEATURES)}"
        )
    return bundle


def input_fn(request_body, request_content_type="application/json"):
    """
    Deserialise a request into a (1, 7) array in training feature order.

    Accepts a single object or a list of objects. Validation happens here
    rather than in predict_fn so a bad request fails before it reaches the
    model, with a message naming the offending field.
    """
    if request_content_type == "text/csv":
        # Positional, in FEATURES order -- documented in the README because
        # there is no header to make it self-describing.
        rows = [
            [float(v) for v in line.split(",")]
            for line in request_body.strip().splitlines()
        ]
        payload = [dict(zip(FEATURES, row)) for row in rows]

    elif request_content_type == "application/json":
        parsed = json.loads(request_body)
        payload = parsed if isinstance(parsed, list) else [parsed]

    else:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Use application/json or text/csv."
        )

    matrix = []
    for i, record in enumerate(payload):
        row = []
        for feature in FEATURES:
            if feature not in record:
                raise ValueError(f"Record {i}: missing feature '{feature}'")
            try:
                value = float(record[feature])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Record {i}: '{feature}' is not numeric "
                    f"(got {record[feature]!r})"
                )
            if not np.isfinite(value):
                raise ValueError(f"Record {i}: '{feature}' must be finite")

            low, high = BOUNDS[feature]
            if not low <= value <= high:
                raise ValueError(
                    f"Record {i}: '{feature}' = {value} is outside the "
                    f"valid range [{low}, {high}]"
                )
            row.append(value)
        matrix.append(row)

    return np.array(matrix, dtype=float)


def predict_fn(input_data, model):
    """
    Apply the exact transform used at training time, then predict.

    Confidence is read by position in `model.classes_`, not by the encoded
    label value. Those coincide only while every class survives into the
    training split; indexing by label value is a latent bug that returns a
    neighbouring class's probability the moment they diverge.
    """
    scaled = model["scaler"].transform(input_data)
    estimator = model["model"]

    indices = estimator.predict(scaled)
    crops = model["encoder"].inverse_transform(indices)

    proba = estimator.predict_proba(scaled)
    classes = list(estimator.classes_)

    results = []
    for i, (index, crop) in enumerate(zip(indices, crops)):
        position = classes.index(index)
        results.append({
            "crop":       str(crop),
            "confidence": round(float(proba[i][position]), 4),
        })
    return results


def output_fn(prediction, accept="application/json"):
    """Serialise the response. A single record returns a single object."""
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    body = prediction[0] if len(prediction) == 1 else prediction
    return json.dumps(body), "application/json"
