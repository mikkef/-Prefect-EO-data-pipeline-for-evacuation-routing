from pathlib import Path

current_file = Path(__file__).resolve()
root_path = Path("/data")

raw_zone_path = root_path / "datalake" / "raw_zone"
silver_zone_path = root_path / "datalake" / "silver_zone"
gold_zone_path = root_path / "datalake" / "gold_zone"

# STAC
STAC_CATALOG_PATH = root_path / "stac_catalog"
STAC_COLLECTION = "stac_catalog/collection.json" # RELATIVE TO S3 BUCKET

# Raw Zone directory paths
# CONFIG_PATH = root_path / "configuration.yaml"
training_aois_dir = raw_zone_path / "training_aois"
inference_aois_dir = raw_zone_path / "inference_aois"
osm_ingested_surfaces_dir = raw_zone_path / "retrieved_surfaces"
sentinel_post_rain_event_data_dir = raw_zone_path / "sentinel2_data" / "sentinel2_post_rain_event_data"
sentinel_annual_raster_median_dir = raw_zone_path / "sentinel2_data" / "annual_pixel_reference"

# Silver Zone directory paths
preprocessed_surfaces_dir = silver_zone_path / "preprocessed_surfaces"
raster_spectral_change_dir = silver_zone_path / "raster_spectral_change"
retrieved_embeddings_dir = raw_zone_path / "retrieved_embeddings"

DATALAKE_HIERARCHY = {
        "raw_zone": [
            "inference_aois",
            "retrieved_embeddings",
            "retrieved_surfaces",
            "training_aois",
        ],
        "silver_zone": [
            "preprocessed_surfaces",
        ],
        "gold_zone": [],
    }