# Deployment

One container serves both the React UI and the FastAPI API on a single port.
That shape is what free container hosts expect, and it means one URL to share.

```
Browser
   │  HTTPS
   ▼
Container (one process, one port)
   ├── /            React bundle (static)
   ├── /docs        OpenAPI UI
   └── /api/v1/*    FastAPI
          ├── validate    Pydantic bounds on all 7 features
          ├── preprocess  scaler.joblib — the exact scaler fitted in training
          ├── predict     RandomForest.joblib (+ 2 comparison models)
          ├── explain     SHAP TreeExplainer, tree_path_dependent
          └── log         JSONL prediction log → drift z-scores
```

The model artifacts are baked into the image. The container **never trains**
and **never reads the dataset** — since the explainer moved to
`tree_path_dependent` SHAP it needs no background sample, so nothing at
serving time touches `data/`.

---

## Choosing a target

| Option | Cost | Card needed | Realistic setup | Verdict |
|---|---|---|---|---|
| **Hugging Face Spaces** | Free | **No** | ~30 min | **Recommended** |
| Google Cloud Run | Free tier covers this | Yes | 2–4 h (+ gcloud install) | Good if you already have GCP billing |
| AWS SageMaker endpoint | ~$0.05+/hr, **no free tier for endpoints** | Yes | 1–2 days | Not worth it here |
| GCP Vertex AI endpoint | Dedicated node, **no free tier** | Yes | 1–2 days | Not worth it here |

**Take Hugging Face Spaces.** It is free with no credit card, gives a public
HTTPS URL, builds from a Dockerfile, and persists. SageMaker and Vertex
endpoints bill by the hour for an idle node — for a portfolio project that
means either a surprise bill or an endpoint you tear down before anyone can
click it.

Deploying somewhere real and being able to explain precisely how it maps to a
managed platform is a stronger position than a half-configured SageMaker
endpoint you cannot afford to leave running.

---

## Deploy to Hugging Face Spaces

**1. Confirm the artifacts exist**

```bash
python deploy/build_space.py --check
```

**2. Create the Space**

<https://huggingface.co/new-space> → SDK **Docker** → template **Blank** →
name it `crop-recommendation`.

**3. Stage and push**

```bash
python deploy/build_space.py
cd .space-build
git init -b main
git add -A
git commit -m "Deploy crop recommendation service"
git remote add space https://huggingface.co/spaces/<your-username>/crop-recommendation
git push --force space main
```

Git will ask for your Hugging Face username and an access token as the
password — create one at <https://huggingface.co/settings/tokens> with
**write** scope.

**4. Watch the build**

The Space's **Logs** tab. First build is ~5–8 minutes: it installs npm
packages, builds the React bundle, then installs the Python dependencies.

**5. Verify**

```bash
curl https://<your-username>-crop-recommendation.hf.space/api/v1/health/ready

curl -X POST https://<your-username>-crop-recommendation.hf.space/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"N":80,"P":40,"K":40,"temperature":23,"humidity":82,"ph":6,"rainfall":200}'
```

Expect `rice`. Then open the URL in a browser and run a prediction through the
dashboard.

**Redeploying** after retraining: re-run `python deploy/build_space.py`, then
`git add -A && git commit -m "..." && git push space main` from `.space-build`.

---

## Alternative: Google Cloud Run

Only if you already have a GCP project with billing enabled. Real GCP, and the
free tier (2M requests/month) covers a portfolio project.

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

python deploy/build_space.py
cd .space-build

gcloud run deploy crop-recommendation \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 7860 \
  --max-instances 3
```

`--memory 2Gi` is not optional: scikit-learn, SHAP and matplotlib together
exceed the 512 MB default and the container will be OOM-killed at startup.
`--max-instances 3` caps the blast radius of a traffic spike on your bill.

---

## Mapping to SageMaker and Vertex AI

Every piece of this project has a managed equivalent. Knowing which piece maps
to which service is the useful knowledge — not the CLI syntax.

| This project | AWS SageMaker | GCP Vertex AI |
|---|---|---|
| `train_pipeline.py` on a laptop | Training Job (`Estimator.fit`) on a managed instance | Custom Training Job |
| `backend/models/*.joblib` on disk | Model artifact in S3 | Model artifact in GCS |
| Committing artifacts to git | Model Registry — versioned, with approval status | Vertex Model Registry |
| `deploy/Dockerfile` | Bring-your-own-container for training or inference | Custom container |
| FastAPI container on a host | Real-time Inference Endpoint | Vertex Endpoint |
| `uvicorn --workers` | Endpoint instance count + autoscaling policy | Endpoint replica count (`min/maxReplicaCount`) |
| `monitoring_baseline.json` + z-scores | SageMaker Model Monitor (baseline job + scheduled monitor) | Vertex Model Monitoring (skew/drift against a training baseline) |
| `logs/predictions.jsonl` | Endpoint Data Capture → S3 | Endpoint request/response logging → BigQuery |
| `retrain.py` promotion gate | Pipeline step + Registry approval before deploy | Vertex Pipelines + Model Registry alias |
| Manual redeploy | Endpoint update with a traffic-shifted variant | Endpoint `trafficSplit` across model versions |
| GitHub Actions | SageMaker Pipelines | Vertex Pipelines (Kubeflow) |

### The concepts worth being able to explain

**Training vs inference infrastructure.** Training is a batch job: it runs
once, wants a lot of CPU or GPU briefly, and is fine to schedule. Inference is
a long-lived service: modest per-request cost, but it must stay up and respond
in milliseconds. That is why managed platforms bill them separately — a
training job stops when it finishes, while an endpoint bills while idle.

**Model artifact vs model registry.** The artifact is the serialised file. The
registry is a versioned catalogue of artifacts with lineage — which data,
which code, which metrics — plus an approval state. `retrain.py` writing
`retrain_history.json` and refusing to promote inside fold noise is a hand-
rolled registry approval gate.

**Rolling out a new model.** Never all at once. Both platforms let you attach
a new version to an existing endpoint and shift a fraction of traffic to it —
a canary. Watch error rate and latency, then shift the rest or roll back. This
project's equivalent is `retrain.py --rollback`, restoring the previous
artifact set.

**Data drift vs concept drift.** Data drift is the *input* distribution
moving: users start submitting soil profiles from a different region. The
drift endpoint here detects that, because it compares recent inputs against
the training baseline. Concept drift is the *relationship* changing: the same
soil now suits a different crop because the climate shifted. **You cannot
detect concept drift without new labels** — no amount of input monitoring
finds it. That is the honest limitation of every monitoring setup that only
watches features, this one included.

**If performance drops.** Check for input drift first, because it is
measurable immediately. Then get labels for recent predictions and measure
real accuracy — that is the only thing that detects concept drift. Roll back
if a recent deploy caused it; retrain if the world moved.
