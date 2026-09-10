# -*- coding: utf-8 -*-
"""
============================================================
deploy/sagemaker/bootstrap_aws.py
============================================================
One-time AWS setup for the SageMaker deployment.

    check credentials -> create execution role -> print the ARN

The execution role is the step that blocks most first deployments.
SageMaker does not run as you; it assumes a role, and that role needs
permission to read your model artifact from S3 and write logs. Doing
it here rather than through the console means it is reproducible and
you can read exactly what permissions were granted.

Idempotent: re-running it finds the existing role and prints the ARN
again rather than failing.

Usage:
  python deploy/sagemaker/bootstrap_aws.py            # check + create
  python deploy/sagemaker/bootstrap_aws.py --check    # check only
  python deploy/sagemaker/bootstrap_aws.py --delete   # remove the role
============================================================
"""

import argparse
import json
import sys

ROLE_NAME = "CropRecommenderSageMakerRole"

# SageMaker must be allowed to assume this role — that is what makes it an
# "execution role" rather than an ordinary one.
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}

# AmazonSageMakerFullAccess is broad. It is the documented starting point and
# is fine for a personal account; for anything shared, scope it down to the
# specific S3 bucket and the SageMaker actions actually used.
MANAGED_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"


def check_credentials():
    """Confirm credentials resolve, and say who they belong to."""
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError

    try:
        identity = boto3.client("sts").get_caller_identity()
    except NoCredentialsError:
        print("[ERROR] No AWS credentials found.")
        print("        Run `aws configure` and enter your access key,")
        print("        or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.")
        return None
    except ClientError as exc:
        print(f"[ERROR] Credentials rejected: {exc.response['Error']['Message']}")
        return None

    print("[AUTH]  Account : " + identity["Account"])
    print("        Identity: " + identity["Arn"])
    return identity


def check_region():
    import boto3
    region = boto3.session.Session().region_name
    if not region:
        print("[WARN]  No default region set. Run `aws configure` and set one,")
        print("        e.g. ap-south-1 (Mumbai) or us-east-1.")
    else:
        print("[AUTH]  Region  : " + region)
    return region


def ensure_role():
    """Create the execution role, or return the existing one."""
    import boto3
    from botocore.exceptions import ClientError

    iam = boto3.client("iam")

    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        print(f"[ROLE]  Already exists: {ROLE_NAME}")
        return role["Arn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            print(f"[ERROR] Could not read role: "
                  f"{exc.response['Error']['Message']}")
            return None

    print(f"[ROLE]  Creating {ROLE_NAME}...")
    try:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="Execution role for the crop recommendation endpoint",
        )["Role"]
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=MANAGED_POLICY)
        print(f"[ROLE]  Attached {MANAGED_POLICY.split('/')[-1]}")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(f"[ERROR] {code}: {exc.response['Error']['Message']}")
        if code in ("AccessDenied", "AccessDeniedException"):
            print("        Your user cannot create IAM roles. Either attach")
            print("        IAMFullAccess to it, or create the role by hand:")
            print("        IAM > Roles > Create role > AWS service > SageMaker")
        return None

    return role["Arn"]


def delete_role():
    import boto3
    from botocore.exceptions import ClientError
    iam = boto3.client("iam")
    try:
        iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn=MANAGED_POLICY)
    except ClientError:
        pass
    try:
        iam.delete_role(RoleName=ROLE_NAME)
        print(f"[ROLE]  Deleted {ROLE_NAME}")
    except ClientError as exc:
        print(f"[SKIP]  {exc.response['Error']['Message']}")


def main():
    parser = argparse.ArgumentParser(description="One-time AWS setup")
    parser.add_argument("--check", action="store_true",
                        help="Verify credentials only; create nothing")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the execution role")
    args = parser.parse_args()

    print("=" * 62)
    print("  AWS bootstrap for the SageMaker deployment")
    print("=" * 62)

    if check_credentials() is None:
        return 1
    check_region()

    if args.delete:
        delete_role()
        return 0

    if args.check:
        print("\n[OK]    Credentials work. Re-run without --check to create "
              "the role.")
        return 0

    arn = ensure_role()
    if arn is None:
        return 1

    print(f"\n[OK]    Execution role ARN:\n        {arn}")
    print("\nNext:")
    print(f"  python deploy/sagemaker/deploy_sagemaker.py --role {arn}")
    print("\nThat deploys a SERVERLESS endpoint: billed per inference-second,")
    print("no charge while idle. Run --teardown when you are finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
