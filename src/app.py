"""
app.py

Prefect orchestration of the surface-classification pipeline.

    training_flow
      1. ingestion_flow                    (waits for BOTH subflows below to succeed)
           |- download-osm-roads           } run concurrently; inside each, one
           |- download-embeddings          } task per AOI, each retried on failure
      2. sample_collection_flow            (starts only after ingestion succeeded)
      3. train-test split -> train models  (start only after sample collection)

    inference_flow
      1. ingestion_flow
      2. one inference task per AOI

Which one runs is decided by `mode` in the YAML config ("training" | "inference").

Run it from this folder:   python app.py
Optional UI:               prefect server start     (in another terminal)
"""
import asyncio
import joblib
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import yaml
from prefect import serve
from prefect import flow, task, get_run_logger
from prefect.deployments import run_deployment
from ingestion.embedding_retriever.app import ingest_embeddings
from ingestion.OSMRetriever.scripts.functions.app import run_OSM_to_IRIDE
from constants import (
    retrieved_embeddings_dir,
    preprocessed_surfaces_dir,
    training_aois_dir,
    inference_aois_dir,
    # CONFIG_PATH,
    STAC_COLLECTION, silver_zone_path
)
import pystac
from pystac.stac_io import DefaultStacIO
from src.samples_gatherer.sample_gatherer_utils import collect_all_training_data
from src.constants import gold_zone_path, DATALAKE_HIERARCHY
from src.common import (
    BUCKET,
    s3_client,
    upload_file_to_bucket,
    upload_shapefile_to_bucket,
)
from train_test.train_test_utils import create_train_test_dict
from train_test.train_test_utils import train_best_model
from inference.inference_utils import infer_raster
from stac_creator.app import update_stac
from botocore.exceptions import ClientError
from src.evacuation_routes_mapping.processing_flow import evacuation_routes_processing
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

current_file = Path(__file__).resolve()
root_path = Path("/data")

LABEL_FIELD = "LABELING"
CLASS_FIELD = "surface_L1"
ALLOWED_CLASSES = ("paved", "unpaved")       # e.g. ("dirt", "gravel")
BUFFER_DISTANCE = 3.0

# "weights" | "downsample_train" | "downsample_all" | "none"
BALANCE = "downsample_all"

# How many top RF features to try in the model comparison
FEATURE_COUNTS = [120]

# The best model is saved here
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.pkl"

# Ingestion: retries and concurrency
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAYS = [10, 30, 60]   # seconds before retry 1, 2, 3
MAX_CONCURRENT_DOWNLOADS = 2           # AOIs downloaded at once PER SOURCE, be polite to the servers

INFERENCE_BUFFER_DISTANCE = 3.0
INFERENCE_THRESHOLD = 0.70

# Object storage settings (endpoint, keys, BUCKET) and the upload functions
# live in src/common.py, shared with the STAC builder.


def _list_aois(aoi_dir) -> list:
    """Accept either a folder of .shp files or an already-built list of paths."""
    if isinstance(aoi_dir, (str, Path)) and Path(aoi_dir).is_dir():
        return sorted(Path(aoi_dir).glob("*.shp"))
    return list(aoi_dir)


# ===========================================================================
# INGESTION = one flow with two subflows (OSM roads, Tessera embeddings)
# ===========================================================================
# The ingestion functions are blocking, so asyncio.to_thread runs each one in a
# worker thread and the event loop stays free to run the other downloads.
# Retries live on the per-AOI tasks: if one AOI fails, only that one is repeated.
_RETRY = dict(
    retries=DOWNLOAD_RETRIES,
    retry_delay_seconds=DOWNLOAD_RETRY_DELAYS,
    retry_jitter_factor=0.5,  # spreads retries out so they do not all hit the server together
)


@task(
    name="osm-surfaces-for-aoi",
    task_run_name=lambda parameters: (
        f"osm-surfaces-{Path(parameters['aoi_shp']).stem}"
    ),
    log_prints=True,
    **_RETRY,
)
async def osm_surfaces_for_aoi(aoi_shp):
    logger = get_run_logger()
    logger.info(f"Starting OSM surfaces download for AOI: {aoi_shp}")
    outfolder = Path(root_path) / "datalake" / "raw_zone" / "retrieved_surfaces"

    return await asyncio.to_thread(run_OSM_to_IRIDE, aoi_shp, outfolder=outfolder, logger=logger)


@task(
    name="embeddings-for-aoi",
    task_run_name=
    lambda parameters: (f"embeddings-{Path(parameters['aoi_shp']).stem}"
    ),
    **_RETRY,)
async def embeddings_for_aoi(aoi_shp):
    logger = get_run_logger()
    logger.info(f"Starting tessera embeddings download for AOI: {aoi_shp}")
    return await asyncio.to_thread(ingest_embeddings, aoi_shp, logger)


async def _run_all(task_fn, aois) -> dict:
    """Run task_fn for every AOI concurrently (bounded). Wait for ALL of them to
    finish, then raise if any failed after its retries."""
    slots = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def one(aoi):
        async with slots:
            return await task_fn(aoi)

    outcomes = await asyncio.gather(*(one(a) for a in aois), return_exceptions=True)
    failed = {a: o for a, o in zip(aois, outcomes) if isinstance(o, BaseException)}
    if failed:
        raise RuntimeError(f"{len(failed)} AOI(s) failed after retries: {failed}")
    return dict(zip(aois, outcomes))


@flow(name="download-osm-surfaces", log_prints=True)
async def download_osm_surfaces(aois) -> dict:
    return await _run_all(osm_surfaces_for_aoi, aois)


@flow(name="download-embeddings", log_prints=True)
async def download_embeddings(aois) -> dict:
    return await _run_all(embeddings_for_aoi, aois)


@flow(name="Ingestion", log_prints=True)
async def ingestion_flow(aoi_list) -> dict:
    """
    Start both download subflows at the same time and wait for both.
    Only if BOTH succeeded does this return; otherwise it raises, and whatever
    awaited it (training / inference) never starts.
    """
    logger = get_run_logger()
    aois = _list_aois(aoi_list)
    logger.info(f"Ingesting {len(aois)} AOI(s)")

    # return_exceptions=True: if one subflow fails, still let the other finish
    # TODO UNCOMMENT
    surfaces, embeddings = await asyncio.gather(
        download_osm_surfaces(aois),
        download_embeddings(aois),
        return_exceptions=True,
    )
    ########################
    logger.info(f"###{surfaces}")
    # TODO UNCOMMENT
    errors = [e for e in (surfaces, embeddings) if isinstance(e, BaseException)]
    if errors:
        raise RuntimeError(f"Ingestion failed: {errors}")
    #######################3

    for aoi in aois:
        # roads are a shapefile (.shp + .dbf/.shx/.prj...), embeddings a single tif
        surfaces_keys = await asyncio.to_thread(
            upload_shapefile_to_bucket, surfaces[aoi], "datalake/raw_zone/retrieved_surfaces"
        )
        emb_key = await asyncio.to_thread(
            upload_file_to_bucket, embeddings[aoi], "datalake/raw_zone/retrieved_embeddings"
        )
        logger.info("Uploaded %s: %d surfaces file(s) and %s", Path(aoi).stem, len(surfaces_keys), emb_key)

    # TODO UNCOMMENT
    return {
        aoi: {"retrieved_surfaces": surfaces[aoi], "embeddings_raster": embeddings[aoi]}
        for aoi in aois
    }


# ===========================================================================
# TRAINING
# ===========================================================================
@task(name="collect-samples")
def collect_samples() -> pd.DataFrame:
    return collect_all_training_data(
        embeddings_dir=retrieved_embeddings_dir,
        roads_dir=preprocessed_surfaces_dir,
        label_field=LABEL_FIELD,
        class_field=CLASS_FIELD,
        buffer_distance=BUFFER_DISTANCE,
        allowed_classes=ALLOWED_CLASSES,
    )


@task(name="train-test-split")
def make_train_test_split(samples_df: pd.DataFrame) -> dict:
    return create_train_test_dict(samples_df, balance=BALANCE)


@task(name="train-models")
def train_models(train_test_splitted_data: dict) -> dict:
    logger = get_run_logger()
    best = train_best_model(train_test_splitted_data, save_path=MODEL_PATH)

    logger.info(
        f"BEST MODEL: {best['best_model_name']} | "
        f"accuracy={best['accuracy']:.4f} | macro F1={best['f1_macro']:.4f} | "
        f"saved to {MODEL_PATH}"
    )
    for class_name, f1 in best["f1_by_class"].items():
        logger.info(f"  F1 {class_name}: {f1:.4f}")
    logger.info(f"Features used ({len(best['selected_features'])}): {best['selected_features']}")

    return {"rf_results": train_test_splitted_data, "best": best}


@flow(name="sample-collection", log_prints=True)
async def sample_collection_flow() -> pd.DataFrame:
    return collect_samples()


@flow(name="training")
async def training_flow(aoi_dir=training_aois_dir):
    # Each line below only starts after the previous one is done (await / direct
    # call both block this flow). If ingestion fails, the exception propagates
    # and nothing after it runs.
    await ingestion_flow(aoi_dir)

    samples_df = await sample_collection_flow()
    split = make_train_test_split(samples_df)
    return train_models(split)


# ===========================================================================
# INFERENCE
# ===========================================================================
@flow(name="infer-aoi")  # no retries: a failure here is a bug or bad data, not a flaky network
async def infer_aoi(aoi, data: dict, best: dict, surface="road"):
    logger = get_run_logger()
    lat, lon, crs = Path(aoi).stem.split("_")[1:4]
    pixel_output_path = gold_zone_path / f"per_{surface}_pixel_inference_{lat}_{lon}_{crs}.tif"
    road_output_path = gold_zone_path / f"per_{surface}_inference_{lat}_{lon}_{crs}.shp"

    infer_raster(
        raster_path=data["embeddings_raster"],
        roads_path=data["retrieved_surfaces"],
        model=best["pipeline"],
        selected_features=best["features"],
        road_output_path=road_output_path,
        pixel_output_path=pixel_output_path,
        buffer_distance=INFERENCE_BUFFER_DISTANCE,
        threshold=INFERENCE_THRESHOLD,
        logger=logger

    )

    # pixel output is a single tif, road output is a shapefile (all its parts go up)
    pixel_key = await asyncio.to_thread(upload_file_to_bucket, pixel_output_path, "datalake/gold_zone")
    surfaces_keys = await asyncio.to_thread(upload_shapefile_to_bucket, road_output_path, "datalake/gold_zone")
    logger.info("Uploaded for AOI %s: %s and %d surfaces file(s)", Path(aoi).stem, pixel_key, len(surfaces_keys))


@flow(name="Create Update STAC")
async def create_update_stac():
    logger = get_run_logger()
    logger.info("Proceeding with STAC catalog update..")
    update_stac(logger)


async def inference_flow(aoi_dir=inference_aois_dir):
    data_dict = await ingestion_flow(aoi_dir)   # waits until all downloads are done

    best = joblib.load(MODEL_PATH)        # load the model once, not once per AOI

    # Sequential on purpose: inference is CPU/RAM heavy. If your machine can take
    # it, use infer_aoi.submit(...) in a loop, as in the ingestion flow.
    for aoi, data in data_dict.items():
        await infer_aoi(aoi, data, best)

    await create_update_stac()



# ===========================================================================

class S3StacIO(DefaultStacIO):
    """Lets pystac read s3://bucket/key files from RustFS via boto3."""

    def read_text(self, source, *args, **kwargs):
        href = str(source)
        if href.startswith("s3://"):
            bucket, key = href[len("s3://"):].split("/", 1)
            obj = s3_client().get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        return super().read_text(source, *args, **kwargs)


def item_bbox_2d(item):
    """Return [west, south, east, north], even if the item has a 3D bbox."""
    b = item.bbox
    return [b[0], b[1], b[3], b[4]] if len(b) == 6 else list(b)


def bbox_contains(outer, inner):
    return (
        outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


async def evacuation_routes_processing_flow():
    await evacuation_routes_processing()
    await asyncio.to_thread(
        upload_shapefile_to_bucket, f"{silver_zone_path}/evacuation_routes_line.shp", "datalake/gold_zone"
    )

    await asyncio.to_thread(
        upload_shapefile_to_bucket, f"{silver_zone_path}/evacuation_routes_point.shp",
        "datalake/gold_zone"
    )


@flow(name="evacuation-routes-calculation")
def evacuation_routes_flow():
    logger = get_run_logger()
    logger.info("Beginning evauation-routes-calculation flow..")
    state = asyncio.run(evacuation_routes_processing_flow())
    if state:
        logger.info("evacuation-routes-calculation flow completed successfully")


@flow(name="embedding-based-surface-classification")
def rem_flow(params: dict = {"bbox": []}):

    logger = get_run_logger()
    s3 = s3_client()

    logger.info(f"root_path = {root_path.resolve()}")
    logger.info(f"working directory = {Path.cwd()}")

    for zone, subdirs in DATALAKE_HIERARCHY.items():
        zone_path = root_path / "datalake" / zone
        zone_path.mkdir(parents=True, exist_ok=True)
        print(f"created {zone_path}")
        for sub in subdirs:
            sub_path = zone_path / sub
            sub_path.mkdir(exist_ok=True)
            print(f"created {sub_path}")

    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
        logger.info(f"Created bucket {BUCKET}")

    bbox = params.get("bbox", [])
    mode = "inference"

    if len(bbox) == 4:
        west, south, east, north = map(float, bbox)

        if not (west < east and south < north):
            raise ValueError("bbox must be [west, south, east, north] in WGS84")
        if not (-180 <= west and east <= 180 and -90 <= south and north <= 90):
            raise ValueError("bbox values are outside WGS84 range")

        requested = [west, south, east, north]

        collection = pystac.Collection.from_file(
            f"s3://{BUCKET}/{STAC_COLLECTION}",
            stac_io=S3StacIO(),
        )

        matched_item = None
        for item in collection.get_items():
            if item.bbox and bbox_contains(item_bbox_2d(item), requested):
                matched_item = item
                break

        if matched_item:
            logger.info(
                "Found STAC item '%s' covering bbox=%s", matched_item.id, requested
            )
            logger.info("Item bbox: %s", matched_item.bbox)
            mode = "existing"

            return


        else:
            logger.info("No STAC item covers bbox=%s. Running inference.", requested)
            mode = "inference"

            west, south, east, north = bbox
            lat = (south + north) / 2
            lon = (west + east) / 2

            out_dir = Path(root_path)/"datalake"/"raw_zone"/"inference_aois"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"aoi_{lat:.5f}_{lon:.5f}_EPSG4326.shp"

            gdf = gpd.GeoDataFrame(
                {"id": [path.stem]},
                geometry=[box(west, south, east, north)],
                crs="EPSG:4326",
            )
            gdf.to_file(path)  # also writes the .shx, .dbf, .prj and .cpg files


    logger.info("Selected mode: %s", mode)
    # with CONFIG_PATH.open("r", encoding="utf-8") as file:
    #     config = yaml.safe_load(file)
    #
    # mode = config.get("mode", "inference").lower()
    # print(f"Mode: {mode} was selected..")

    if mode == "training":
        return asyncio.run(training_flow())
    elif mode == "inference":
        return asyncio.run(inference_flow())

    raise ValueError(f"Invalid mode '{mode}'. Expected 'training' or 'inference'.")


@flow(name="main_flow")
def main_flow(params: dict = {"bbox": []}):
    rem_run = run_deployment(
        name=f"{rem_flow.name}/serve",
        parameters={"params": params},   # match rem_flow's real signature
        timeout=None,                    # wait until it finishes
    )
    if not rem_run.state.is_completed():
        raise RuntimeError(f"rem_flow ended in {rem_run.state.name}; evacuation not started")

    run_deployment(
        name=f"{evacuation_routes_flow.name}/serve",
        timeout=None,
    )

if __name__ == "__main__":
    serve(
        main_flow.to_deployment(name="serve"),
        rem_flow.to_deployment(name="serve"),
        evacuation_routes_flow.to_deployment(name="serve"),
    )