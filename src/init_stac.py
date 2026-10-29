import json
import os

import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    region_name=os.environ.get("S3_REGION", "us-east-1"),
)

bucket = os.environ.get("S3_BUCKET", "portofolio")
key = "stac_catalog/collection.json"

collection = {
    "type": "Collection",
    "stac_version": "1.1.0",
    "id": "stac_collection",
    "description": "STAC collection for geospatial products.",
    "license": "proprietary",
    "extent": {
        "spatial": {
            "bbox": [[-180.0, -90.0, 180.0, 90.0]]
        },
        "temporal": {
            "interval": [[None, None]]
        },
    },
    "links": [
        {
            "rel": "self",
            "href": "./collection.json",
            "type": "application/json",
        },
        {
            "rel": "root",
            "href": "./collection.json",
            "type": "application/json",
        },
    ],
}

s3.put_object(
    Bucket=bucket,
    Key=key,
    Body=json.dumps(collection, indent=2).encode("utf-8"),
    ContentType="application/json",
)

print(f"Created STAC Collection: s3://{bucket}/{key}")

