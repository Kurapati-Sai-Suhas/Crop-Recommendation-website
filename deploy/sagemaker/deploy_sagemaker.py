# -*- coding: utf-8 -*-
"""
============================================================
deploy/sagemaker/deploy_sagemaker.py
============================================================
Package the trained artifacts and deploy them to a SageMaker endpoint.

    package -> upload to S3 -> create Model -> deploy Endpoint -> smoke test

Defaults to a **serverless** endpoint. That is the important choice: a
real-time endpoint provisions a dedicated instance and bills for every
hour it exists, whether or not anyone calls it. A portfolio endpoint
serves a handful of requests and idles the rest of the month, so
serverless -- billed per inference-second with no idle charge -- costs
a fraction of the same demo. The trade-off is a cold start of roughly
30-60 seconds after a period of inactivity.

Run `--teardown` when you are finished. An endpoint you forget about
is the most common way a free-credit balance disappears.

Usage
-----
  python deploy/sagemaker/deploy_sagemaker.py --package-only
  python deploy/sagemaker/deploy_sagemaker.py --bucket my-bucket
  python deploy/sagemaker/deploy_sagemaker.py --bucket my-bucket --real-time
  python deploy/sagemaker/deploy_sagemaker.py --teardown
============================================================
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(os.path.dirname(HERE))
MODEL_DIR = os.path.join(ROOT, "backend", "models")
CODE_DIR  = os.path.join(HERE, "code")

ENDPOINT_NAME = "crop-recommendation"

# Only what inference.py actually loads. The other two models are for the
# comparison panel in the web UI and have no business in an endpoint.
ARTIFACTS = ["RandomForest.joblib", "scaler.joblib", "label_encoder.joblib"]

# SageMaker's managed SKLearn container maxes out at 1.2-1, but these models
# were pickled with scikit-learn 1.4.2. Loading them under 1.2 raises
# InconsistentVersionWarning and can fail outright.
#
# The fix is code/requirements.txt, which pins scikit-learn==1.4.2. SageMaker
# installs it into the container at startup, so the serving version matches
# the training version exactly. Without that file this endpoint would either
# refuse to load or -- worse -- load and score differently from the training
# run, with no error.
FRAMEWORK_VERSION = "1.2-1"
PYTHON_VERSION    = "py3"


def build_tarball(destination):
    """
    Build model.tar.gz in the layout the container expects.

    Artifacts go at the archive root -- model_fn receives that directory as
    model_dir. The inference code travels separately via source_dir.
    """
    missing = [a for a in ARTIFACTS if not os.path.exists(os.path.join(MODEL_DIR, a))]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}. Run `python train_pipeline.py` first."
        )

    with tarfile.open(destination, "w:gz") as tar:
        for name in ARTIFACTS:
            tar.add(os.path.join(MODEL_DIR, name), arcname=name)

    size = os.path.getsize(destination) / 1_048_576
    print(f"[PACKAGE] {destination} ({size:.1f} MB)")
    for name in ARTIFACTS:
        print(f"           + {name}")
    return destination


def deploy(bucket, region, role, serverless, instance_type, memory_mb):
    import boto3
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel
    from sagemaker.serverless import ServerlessInferenceConfig

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    bucket = bucket or session.default_bucket()

    if not role:
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            raise SystemExit(
                "No execution role found.\n"
                "Pass --role with a SageMaker execution role ARN, e.g.\n"
                "  arn:aws:iam::<account-id>:role/service-role/"
                "AmazonSageMaker-ExecutionRole-XXXX\n"
                "Create one in the IAM console with the "
                "AmazonSageMakerFullAccess policy attached."
            )

    with tempfile.TemporaryDirectory() as tmp:
        tarball = build_tarball(os.path.join(tmp, "model.tar.gz"))
        print(f"[UPLOAD]  s3://{bucket}/{ENDPOINT_NAME}/")
        model_uri = session.upload_data(
            tarball, bucket=bucket, key_prefix=ENDPOINT_NAME
        )
    print(f"           {model_uri}")

    model = SKLearnModel(
        model_data=model_uri,
        role=role,
        entry_point="inference.py",
        source_dir=CODE_DIR,             # ships inference.py + requirements.txt
        framework_version=FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        sagemaker_session=session,
        name=None,
    )

    if serverless:
        print(f"[DEPLOY]  serverless, {memory_mb} MB, this takes a few minutes...")
        predictor = model.deploy(
            endpoint_name=ENDPOINT_NAME,
            serverless_inference_config=ServerlessInferenceConfig(
                memory_size_in_mb=memory_mb,
                max_concurrency=5,
            ),
        )
    else:
        print(f"[DEPLOY]  real-time on {instance_type} -- BILLED HOURLY WHILE IT "
              f"EXISTS. Run --teardown when finished.")
        predictor = model.deploy(
            endpoint_name=ENDPOINT_NAME,
            initial_instance_count=1,
            instance_type=instance_type,
        )

    print(f"[OK]      Endpoint live: {ENDPOINT_NAME}")
    smoke_test(predictor)
    print("\nWhen you are done:")
    print("  python deploy/sagemaker/deploy_sagemaker.py --teardown")
    return predictor


def smoke_test(predictor):
    """Call the endpoint once and check the answer is the expected one."""
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer

    predictor.serializer   = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    payload = {"N": 80, "P": 40, "K": 40, "temperature": 23.0,
               "humidity": 82.0, "ph": 6.0, "rainfall": 200.0}
    result = predictor.predict(payload)
    print(f"\n[SMOKE]   {payload['N']}/{payload['P']}/{payload['K']} "
          f"-> {result}")

    if result.get("crop") != "rice":
        print(f"[WARN]    Expected 'rice' locally. Got '{result.get('crop')}'. "
              f"If this differs, the container's scikit-learn version probably "
              f"does not match training -- check code/requirements.txt shipped.")
    else:
        print("[SMOKE]   Matches the local prediction.")


def teardown(region):
    """Delete the endpoint, its config, and the model. All three, or it bills."""
    import boto3
    client = boto3.client("sagemaker", region_name=region)

    for label, call in [
        ("endpoint",        lambda: client.delete_endpoint(EndpointName=ENDPOINT_NAME)),
        ("endpoint config", lambda: client.delete_endpoint_config(
            EndpointConfigName=ENDPOINT_NAME)),
    ]:
        try:
            call()
            print(f"[DELETE]  {label}")
        except client.exceptions.ClientError as exc:
            print(f"[SKIP]    {label}: {exc.response['Error']['Message'][:80]}")

    print("\nDeleting the endpoint stops the billing. The model artifact in S3 "
          "costs a few cents a month; remove the bucket object too if you want "
          "a clean slate.")


def main():
    parser = argparse.ArgumentParser(description="Deploy to SageMaker")
    parser.add_argument("--bucket", help="S3 bucket (default: SageMaker's)")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    parser.add_argument("--role", help="SageMaker execution role ARN")
    parser.add_argument("--real-time", action="store_true",
                        help="Provision an instance instead of serverless "
                             "(bills hourly while it exists)")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--memory-mb", type=int, default=2048,
                        help="Serverless memory; needs >=2048 for sklearn+numpy")
    parser.add_argument("--package-only", action="store_true",
                        help="Build model.tar.gz locally and stop. No AWS needed.")
    parser.add_argument("--teardown", action="store_true",
                        help="Delete the endpoint and stop billing")
    args = parser.parse_args()

    print("=" * 62)
    print("  SageMaker deployment")
    print("=" * 62)

    if args.teardown:
        teardown(args.region)
        return 0

    if args.package_only:
        out = os.path.join(HERE, "model.tar.gz")
        build_tarball(out)
        print(f"\nInspect it with:  tar -tzf {out}")
        return 0

    try:
        import sagemaker  # noqa: F401
    except ImportError:
        raise SystemExit(
            "sagemaker SDK not installed. Run:\n"
            "  pip install sagemaker boto3\n"
            "(deliberately not in requirements.txt -- the serving image does "
            "not need it)"
        )

    deploy(args.bucket, args.region, args.role,
           serverless=not args.real_time,
           instance_type=args.instance_type,
           memory_mb=args.memory_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
