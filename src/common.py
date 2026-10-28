"""
common.py

Shared object-storage helpers (RustFS / Silo / MinIO / any S3-compatible server).
Both the pipeline (app.py) and the STAC builder (stac_utils.py) import from here, so
the endpoint, credentials and bucket name are defined in ONE place.

    upload_file_to_bucket(path, prefix)        -> one file (tif, gpkg, json, ...) -> key
    upload_shapefile_to_bucket(shp, prefix)    -> .shp + .dbf/.shx/.prj/... -> [keys]

Both keep the file name and put it under "<prefix>/", for example
    upload_file_to_bucket("x/embeddings_1.tif", "datalake/raw_zone/retrieved_embeddings")
    -> "datalake/raw_zone/retrieved_embeddings/embeddings_1.tif"

Both accept skip_if_same_size=True: an object that already exists with the same size is
left alone (saves re-uploading big rasters).
"""
from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import os
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Settings: the only place where these are defined
# --------------------------------------------------------------------------
ENDPOINT = os.environ.get(
    "S3_ENDPOINT_URL",
    "http://127.0.0.1:9000"
)
ACCESS_KEY = os.environ.get("RUSTFS_ACCESS_KEY", "admin")
SECRET_KEY = os.environ.get("RUSTFS_SECRET_KEY", "admin")
BUCKET = os.environ.get("S3_BUCKET", "portofolio")


# --------------------------------------------------------------------------
# Client / bucket
# --------------------------------------------------------------------------
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def _is_not_found(err: ClientError) -> bool:
    code = str(err.response.get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound", "NoSuchBucket")


def ensure_bucket() -> None:
    """Create the bucket if it does not exist yet."""
    s3 = s3_client()
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as err:
        if not _is_not_found(err):
            raise
        s3.create_bucket(Bucket=BUCKET)
        logger.info("Created bucket %s", BUCKET)


def key_exists(key: str) -> bool:
    try:
        s3_client().head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as err:
        if _is_not_found(err):
            return False
        raise  # e.g. access denied: do not pretend the key is missing


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
def upload_file(local_path, key: str, skip_if_same_size: bool = False) -> bool:
    """Upload one file to an exact key. Returns True if it was uploaded,
    False if it was skipped because the same size is already there."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Cannot upload, file not found: {local_path}")

    s3 = s3_client()
    if skip_if_same_size:
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head["ContentLength"] == local_path.stat().st_size:
                return False
        except ClientError as err:
            if not _is_not_found(err):
                raise
    s3.upload_file(str(local_path), BUCKET, key)
    return True


def upload_file_to_bucket(local_path, prefix: str, skip_if_same_size: bool = False) -> str:
    """Upload a single file (tif, gpkg, ...) to '<prefix>/<file name>'. Returns the key."""
    local_path = Path(local_path)
    key = f"{prefix.strip('/')}/{local_path.name}"
    upload_file(local_path, key, skip_if_same_size)
    return key


def shapefile_parts(shp_path) -> list[Path]:
    """The .shp and every sibling with the same name (.dbf, .shx, .prj, .cpg, ...)."""
    shp_path = Path(shp_path)
    if not shp_path.is_file():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    return sorted(p for p in shp_path.parent.glob(f"{shp_path.stem}.*") if p.is_file())


def upload_shapefile_to_bucket(shp_path, prefix: str, skip_if_same_size: bool = False) -> list[str]:
    """Upload a shapefile with all its parts to '<prefix>/<name>.<ext>'. Returns the keys.
    A .shp alone is unusable, so the sibling files always go with it."""
    return [upload_file_to_bucket(p, prefix, skip_if_same_size) for p in shapefile_parts(shp_path)]


# --------------------------------------------------------------------------
# Download / keys
# --------------------------------------------------------------------------
def download_prefix(prefix: str, local_dir) -> int:
    """Download every object under '<prefix>/' into local_dir, keeping the structure.
    Returns the number of files downloaded."""
    s3 = s3_client()
    prefix = prefix.rstrip("/")
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1:]
            if not rel:
                continue
            target = Path(local_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, obj["Key"], str(target))
            count += 1
    return count


def bucket_key(path) -> str:
    """The part of a local path from its 'datalake' folder onwards, as a bucket key.

    E:/.../tessera_approach/datalake/gold_zone/x.tif -> 'datalake/gold_zone/x.tif'
    Works for folders too, so bucket_key(file.parent) gives an upload prefix.
    """
    parts = Path(path).resolve().parts
    idx = [i for i, p in enumerate(parts) if p == "datalake"]
    if not idx:
        raise ValueError(f"{path} is not inside a 'datalake' folder, cannot build a bucket key")
    return "/".join(parts[idx[-1]:])