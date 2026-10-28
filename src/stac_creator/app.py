from src.stac_creator.stac_utils import read_run_attributes, update_catalog
from src.constants import gold_zone_path, STAC_CATALOG_PATH
from stac_creator.stac_utils import RunAttributes


def check_missing_files(logger, runs):
    # Check if the result files were created
    missing = {}

    for r in runs:
        missing_files_per_run = [
            n
            for n, v in [
                ("embeddings", r.embeddings_file),
                ("roads", r.roads_file),
            ]
            if v is None
        ]

        logger.info(
            f"  {r.key}: outputs={sorted(r.gold_files)} "
            f"missing={missing_files_per_run or None}"
        )

        if missing_files_per_run:
            missing[r.key] = missing_files_per_run

    # Handle missing outputs outside the loop
    if missing:
        raise FileNotFoundError(
            f"Missing required outputs: {missing}"
        )


def update_stac(logger, only_keys=None, use_bucket=True):
    """Add runs to the STAC catalog.

    only_keys : set of run keys to add, e.g. {"39.88243_20.74780_EPSG4326"}.
                None = every run found in gold_zone (slow: every file is hashed).
    use_bucket: True  = read the catalog from the bucket, save locally, upload back.
                False = local only.
    """
    runs = read_run_attributes(only_keys=only_keys)
    logger.info(f"Found {len(runs)} run(s) in {gold_zone_path}")
    if not runs:
        logger.warning("Nothing to add to the catalog")
        return None

    check_missing_files(logger, runs)

    col = update_catalog(runs, validate=False, use_bucket=use_bucket, log=logger)
    logger.info(f"Saved catalog to {STAC_CATALOG_PATH / 'collection.json'}")
    return col