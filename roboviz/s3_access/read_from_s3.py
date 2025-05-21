import boto3
from botocore.exceptions import ClientError
import sys
import os
import json
from pathlib import Path

"""
Downloads the dataset from remote_file_path to specified local_file_path
"""
def download_dataset(endpoint_url, bucket_name, remote_file_path, local_file_path):
    # --- 1. Load credentials from JSON ------------------------------------------
    CREDS_FILENAME = "kopah_creds.json"

    creds_path = Path(__file__).resolve().parents[3] / CREDS_FILENAME

    if not creds_path.is_file():
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")

    creds = json.loads(creds_path.read_text())

    # --- 2. Create the resource --------------------------------------------------
    s3 = boto3.resource(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
    )

    s3_obj = s3.Object(bucket_name, remote_file_path)
    # download hdf5
    if not os.path.exists(local_file_path):
        print("Downloading file")
        try:
            with open(local_file_path, "wb") as f:
                s3_obj.download_fileobj(f)
        except ClientError as e:
            print(e)
