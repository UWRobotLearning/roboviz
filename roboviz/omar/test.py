import os
import glob
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from scipy.spatial.transform import Rotation as R
import boto3
import sys
import subprocess
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

def upload_plots(directory: str,
                 bucket: str,
                 dataset_name: str,
                 endpoint_url: str | None = None,
                 creds_json: str | None = None):
    """
    Walk `directory`, find every *.html file, and upload it to
    s3://<bucket>/<dataset_name>/plots/<relative_path>.html

    Parameters
    ----------
    directory : str
        Local path containing Plotly HTML files (can have sub-folders).
    bucket : str
        Name of the S3 bucket.
    dataset_name : str
        Top-level prefix in the bucket.
    endpoint_url : str | None
        Custom S3 endpoint (leave None for AWS).
    creds_json : str | None
        Optional path to JSON creds file with
        { "aws_access_key_id": "...", "aws_secret_access_key": "..." }.
    """
    print("Uploading")
    # ----------- build the S3 client -----------
    if creds_json:
        with open(creds_json) as fh:
            creds = json.load(fh)
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=creds["access_key"],
            aws_secret_access_key=creds["secret_key"],
        )
    else:
        s3 = boto3.client("s3", endpoint_url=endpoint_url)

    prefix = f"{dataset_name}/plots"           # destination “folder” in the bucket

    # ----------- walk local dir & upload -----------
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(".html"):
                continue                        # skip non-HTML files (optional)

            local_path = os.path.join(root, fname)
            print(local_path)
            # strip off the base directory so we preserve any nested structure
            rel_path = os.path.relpath(local_path, directory)

            # build the full object key and normalize separators
            s3_key = os.path.join(prefix, rel_path).replace(os.sep, "/")

            s3.upload_file(local_path, bucket, s3_key)
            
            print(f"✓ {local_path} → s3://{bucket}/{s3_key}")


if __name__ == "__main__":
    upload_plots("/home/marco/Roboviz/roboviz/roboviz/app/static", 'roboviz-dataset', 'Test', "https://s3.kopah.uw.edu", "/home/marco/Roboviz/kopah_creds.json")