# SageMaker deployment

Deploys the trained Random Forest to a SageMaker endpoint, as an alternative
to the single-container deployment in [`../DEPLOY.md`](../DEPLOY.md).

> **Status: written and locally verified, not deployed.** The inference
> handlers are tested against the live artifacts and produce predictions
> identical to the FastAPI service. The packaging step runs. The AWS steps
> have not been executed — they need credentials.

---

## Use serverless, not a real-time endpoint

This is the decision that matters, and it is the default here.

| | Serverless | Real-time |
|---|---|---|
| Billing | per inference-second | **every hour the endpoint exists** |
| Idle cost | none | full hourly rate |
| Cold start | ~30–60 s after inactivity | none |
| Right for | a portfolio demo | sustained production traffic |

A portfolio endpoint serves a handful of requests and idles the rest of the
month. A real-time endpoint bills for all of that idle time. **Forgetting to
delete a real-time endpoint is the single most common way a free-credit
balance disappears.**

```bash
python deploy/sagemaker/deploy_sagemaker.py --teardown
```

Run that when you are finished. It deletes the endpoint *and* its config —
both are needed to stop billing.

---

## The version trap

SageMaker's managed scikit-learn container tops out at **1.2-1**. These models
were pickled with **scikit-learn 1.4.2**. Loading a 1.4.2 pickle under 1.2
raises `InconsistentVersionWarning` and can fail outright — or, worse, load
and score differently from training with no error at all.

The fix is [`code/requirements.txt`](code/requirements.txt), which pins
`scikit-learn==1.4.2`. SageMaker installs it into the container at startup, so
the serving version matches the training version exactly.

This is worth understanding rather than copying: **any** managed inference
container has a fixed framework version, and silently scoring under a
different one than you trained on is a real production failure mode.

---

## Prerequisites

```bash
pip install sagemaker boto3
```

Deliberately not in `requirements.txt` — the serving image does not need it.

You also need:

1. **AWS credentials** — `aws configure`, or environment variables.
2. **A SageMaker execution role.** IAM → Roles → Create role → SageMaker →
   attach `AmazonSageMakerFullAccess`. Copy the ARN; pass it as `--role`.
   This is the step that most often blocks a first deployment.
3. **A region.** Defaults to `ap-south-1` (Mumbai). Serverless inference is
   not available in every region — check before assuming.

---

## Deploy

```bash
# 1. Check the packaging works. No AWS needed.
python deploy/sagemaker/deploy_sagemaker.py --package-only

# 2. Deploy (serverless by default)
python deploy/sagemaker/deploy_sagemaker.py \
  --role arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX

# 3. When finished — do not skip this
python deploy/sagemaker/deploy_sagemaker.py --teardown
```

The script builds `model.tar.gz`, uploads it to S3, creates the model, deploys
the endpoint, and runs a smoke test asserting the classic rice profile still
returns `rice`. If it does not, the container's scikit-learn version is the
first thing to check.

---

## Calling the endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")
response = runtime.invoke_endpoint(
    EndpointName="crop-recommendation",
    ContentType="application/json",
    Body=json.dumps({"N": 80, "P": 40, "K": 40, "temperature": 23.0,
                     "humidity": 82.0, "ph": 6.0, "rainfall": 200.0}),
)
print(json.loads(response["Body"].read()))
# {"crop": "rice", "confidence": 0.86}
```

Accepts a single object, a list of objects for batch, or `text/csv` with
values **positionally** in `N,P,K,temperature,humidity,ph,rainfall` order —
there is no header, so the order is the contract.

---

## What the handlers do, and why they are overridden

The container supplies defaults for everything except `model_fn`. All four are
overridden here:

| Handler | Why not the default |
|---|---|
| `model_fn` | Loads three artifacts as a bundle — scaler, forest, encoder. Keeping them together makes it impossible to load a model without the transform that belongs to it. |
| `input_fn` | The default hands a raw array straight to the model, **skipping scaling entirely** and returning confident nonsense. It also rebuilds the feature vector explicitly, because a JSON object has no inherent order and the model was fitted on a fixed one. |
| `predict_fn` | Applies scaler → forest → label encoder in order, and reads confidence by **position in `model.classes_`**, not by encoded label value. Those coincide only while every class survives into the training split. |
| `output_fn` | Returns crop and confidence rather than a bare class index. |

Validation is repeated here even though the FastAPI service already does it:
a SageMaker endpoint can be called directly, bypassing the API, and an
endpoint that accepts anything will happily score nonsense.

---

## Verified locally

Handlers were exercised against the live artifacts without AWS:

- `model_fn` loads all three artifacts; 22 classes
- single JSON, batch JSON, and CSV inputs all return correct predictions
- swapping N and rainfall changes the prediction — feature order is respected
- **predictions match the FastAPI service exactly** across three profiles
  (`rice` 0.8600, `mothbeans` 0.5950, `apple` 0.7100)
- validation rejects missing features, out-of-range values, non-numeric
  values, and unsupported content types
- `--package-only` produces a 0.8 MB `model.tar.gz` with the artifacts at the
  archive root

Not verified: the S3 upload, model creation, endpoint deployment, and the
live smoke test. Those need credentials.
