# SageMaker deployment

Deploys the trained Random Forest to a SageMaker endpoint, as an alternative
to the single-container deployment in [`../DEPLOY.md`](../DEPLOY.md).

> **Status: deployed and verified.** Live at endpoint `crop-recommendation`
> in `us-east-1`, serverless, 3072 MB, max concurrency 5. Predictions match
> the local FastAPI service exactly across four profiles. Warm latency
> ~400–450 ms; first call after idle ~1.6 s.
>
> **Remember to run `--teardown` when you no longer need it.**

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

## The version trap, and how it was actually solved

SageMaker's managed scikit-learn container tops out at **1.2-1** — Python 3.9
with scikit-learn 1.2.1. This project trains on **1.4.2**. A 1.4.2 pickle
cannot be relied on in that container.

The obvious fix is a `requirements.txt` in `source_dir` so the container
pip-installs 1.4.2 at startup. **That was tried and it failed twice, in two
different ways.** Both are worth knowing, because both are generic to managed
inference containers:

**Attempt 1 — pinned `scikit-learn` and `numpy`, not `scipy`.** Upgrading
numpy underneath the container's pre-compiled scipy broke scipy's C
extensions:

```
scipy/interpolate/_fitpack_impl.py: array([], dfitpack_int)
-> sagemaker_containers._errors.ClientError
-> /ping returns 500
```

The endpoint failed with *"Unable to stand up your model within the allotted
180 second timeout"*, which reads like a performance problem. It was not: the
model never loaded at all. **The error message pointed at the wrong thing —
only CloudWatch showed the real cause.**

**Attempt 2 — pinned all three.** The model loaded and the endpoint reached
`InService`, but every invocation returned 500:

```
RandomForestClassifier.predict_proba -> Cython
ClientError: Cannot convert numpy.ndarray to numpy.ndarray
```

That message is the signature of **two numpy builds in one process** — a
compiled extension receiving arrays created by a different numpy than it was
built against.

**The fix: stop fighting the image.** Rather than installing a different
scientific stack into a container built around one,
[`build_container_native_model.py`](build_container_native_model.py) rebuilds
the model under the container's own versions (Python 3.9, scikit-learn 1.2.1,
numpy 1.23.5) and writes it to `artifacts/`. `code/requirements.txt` is gone;
**nothing is installed at runtime.**

The rebuilt model scores **0.9955 — delta 0.0000** against the 1.4.2 model.
Same data, same seed, same hyperparameters: the same model expressed under an
older library, not an approximation. `artifacts/provenance.json` records
exactly which versions built it.

Two side benefits: the cold start no longer has to fit a 70 MB pip install
inside the 180-second budget, and the endpoint can never silently score under
a different library than the one it was built with.

---

## Prerequisites

```bash
pip install "sagemaker<3" boto3
```

Deliberately not in `requirements.txt` — the serving image does not need it.

**The `<3` pin is required.** SageMaker Python SDK v3 removed the
`sagemaker.sklearn` module, so `from sagemaker.sklearn.model import
SKLearnModel` raises `ModuleNotFoundError` on v3. `pip install sagemaker`
gives you v3 by default. Verified against v2.257.6, which prints a
deprecation warning on import — silence it with
`SAGEMAKER_SUPPRESS_V2_WARNING=1` if it is noisy.

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
# 0. Verify credentials and create the execution role (once)
python deploy/sagemaker/bootstrap_aws.py

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

## Verified against the live endpoint

Four profiles invoked through `boto3` and compared against the local service:

| Profile | Endpoint | Local | Latency |
|---|---|---|---|
| 80/40/40, humid, 200 mm | `rice` 0.8600 | `rice` 0.8600 | 1565 ms (cold) |
| 10/10/10, dry, pH 4.2 | `mothbeans` 0.5950 | `mothbeans` 0.5950 | 396 ms |
| 120/130/200, humid | `apple` 0.7100 | `apple` 0.7100 | 454 ms |
| 20/135/200, humid | `apple` 1.0000 | `apple` 1.0000 | 439 ms |

All four match to the fourth decimal. Batch (a JSON list) and `text/csv` input
both return correct results. Out-of-range input is rejected — though note it
surfaces as a `ModelError` 500 rather than a clean 400, because SageMaker
wraps any handler exception that way. The validation works; the status code
is less informative than the API's 422.

## Verified locally, before deploying

Handlers were exercised against the artifacts without AWS:

- `model_fn` loads all three artifacts; 22 classes
- single JSON, batch JSON, and CSV inputs all return correct predictions
- swapping N and rainfall changes the prediction — feature order is respected
- **predictions match the FastAPI service exactly** across three profiles
  (`rice` 0.8600, `mothbeans` 0.5950, `apple` 0.7100)
- validation rejects missing features, out-of-range values, non-numeric
  values, and unsupported content types
- `--package-only` produces a 0.8 MB `model.tar.gz` with the artifacts at the
  archive root

All of these were checked before the first deploy attempt, which is why the
failures that followed were all environmental rather than logic errors in the
handlers.
