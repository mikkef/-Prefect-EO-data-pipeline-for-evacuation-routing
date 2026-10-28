import os
import boto3
from botocore.config import Config

LOCAL_STAC_DIR = r"E:/Programming/SIG/surface_type_feasibility_analysis/tessera_approach/datalake"   # the folder that holds collection.json
PREFIX = "datalake"
ENDPOINT = "http://127.0.0.1:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "admin"
BUCKET = "tessera"

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

if __name__ == '__main__':
    s3 = s3_client()
    for root, _, files in os.walk(LOCAL_STAC_DIR):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, LOCAL_STAC_DIR).replace("\\", "/")
            s3.upload_file(local_path, BUCKET, f"{PREFIX}/{rel}")
            print("uploaded", f"{PREFIX}/{rel}")