"""
Build / update a static STAC catalog (collection.json + one Item per run) from the
medallion-style folders:

    datalake/
    ├── gold_zone/                       per_road_inference_{lat}_{lon}_{crs}.*
    │                                    per_pixel_inference_{lat}_{lon}_{crs}.*
    ├── raw_zone/retrieved_embeddings/   tessera_embeddings_{lat}_{lon}_{crs}.*
    └── silver_zone/preprocessed_roads/  roads_{lat}_{lon}_{crs}_02_edit.*

One run = one (lat, lon, crs) key = one STAC Item. The Item holds up to four assets:
per-road inference, per-pixel inference, the embeddings used, and the OSM roads used.

Only "primary" files (.tif, .shp, ...) are treated as data. Sidecar files such as
.tif.aux.xml, .shp.xml, .dbf, .shx, .prj, .cpg are ignored.

How the update works (update_catalog):
    1. If the bucket already has catalog/collection.json, the whole catalog/ prefix is
       downloaded into STAC_CATALOG_PATH and loaded. (If the bucket has none but a local
       collection.json exists, that one is loaded. Otherwise a new collection is created.)
    2. The items of the runs given are added. An item with the same id is replaced.
    3. The catalog is ALWAYS written locally (STAC_CATALOG_PATH).
    4. Then it is uploaded: data files first, item JSONs next, collection.json last.

Requires: pystac, rasterio, boto3 (via src/common.py)  (optional: geopandas for vector bbox fallback)
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pystac
from pystac.extensions.projection import ProjectionExtension
from src.common import (
    BUCKET,
    bucket_key,
    download_prefix,
    ensure_bucket,
    key_exists,
    upload_file,
    upload_file_to_bucket,
    upload_shapefile_to_bucket,
)
from src.constants import gold_zone_path, osm_ingested_surfaces_dir, retrieved_embeddings_dir, STAC_CATALOG_PATH

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration: edit these
# --------------------------------------------------------------------------
REPO_ROOT = Path(".").resolve()
DATALAKE = REPO_ROOT / "datalake"

# Object storage settings (endpoint, keys, bucket) live in src/common.py
CATALOG_PREFIX = "stac_catalog"                          # where the catalog lives in the bucket
COLLECTION_KEY = f"{CATALOG_PREFIX}/collection.json"


# The imagery / embeddings source date, the one variable you mentioned.
EMBEDDINGS_SRC_DATE = datetime(2026, 3, 12, tzinfo=timezone.utc)

PIPELINE_NAME = "embedding-based-surface-classification"
PIPELINE_VERSION = "0.2.0"
EMBEDDING_MODEL = "tessera-foundation-model-v1"
METHOD = "pixel_classifier_vote"

COLLECTION_ID = "embedding-based-surface-classification"
PREFIX = "EBSC"  # prefix for our custom properties

# Only these extensions count as real data files, in order of preference.
# Everything else (.aux.xml, .shp.xml, .dbf, .shx, .prj, .cpg, ...) is a sidecar.
PRIMARY_EXTS = (".tif", ".tiff", ".shp", ".gpkg", ".parquet", ".geojson")

# Fallback only: if the real extent cannot be read from the file, build a box
# of +/- this many degrees around the (lat, lon) in the file name.
FALLBACK_HALF_SIZE_DEG = 0.05

# Hashing big rasters is slow. Turn off if you do not need file:checksum.
COMPUTE_CHECKSUMS = True

PROCESSING_EXT = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
FILE_EXT = "https://stac-extensions.github.io/file/v2.1.0/schema.json"

# --------------------------------------------------------------------------
# Data structure
# --------------------------------------------------------------------------
# crs is letters/digits only (e.g. EPSG4326): no dots, so "EPSG4326.tif.aux"
# can never be captured as a CRS.
GOLD_PATTERN = re.compile(
    r"^(?P<kind>per_road|per_road_pixel)_inference_"
    r"(?P<lat>-?\d+(?:\.\d+)?)_(?P<lon>-?\d+(?:\.\d+)?)_(?P<crs>[A-Za-z0-9]+)$"
)


@dataclass
class RunAttributes:
    """Everything we know about one run, collected from the file system."""

    lat: str  # kept as the raw string from the file name, so paths match exactly
    lon: str
    crs: str
    epsg: int | None
    bbox: list[float]  # WGS84 [west, south, east, north]
    datetime_: datetime  # what the data represents (embeddings source date)
    created: datetime  # when the inference was produced
    osm_snapshot: datetime | None  # None if no OSM was used
    gold_files: dict[str, Path] = field(default_factory=dict)  # kind -> path
    embeddings_file: Path | None = None
    roads_file: Path | None = None

    @property
    def key(self) -> str:
        return f"{self.lat}_{self.lon}_{self.crs}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _is_primary(path: Path) -> bool:
    """True for real data files, False for sidecars like .tif.aux.xml or .dbf."""
    return path.is_file() and path.suffix.lower() in PRIMARY_EXTS


def _find(folder: Path, stem: str) -> Path | None:
    """Find '<stem><primary extension>' in folder (exact name, no globbing).

    Returns None if missing. Extensions are tried in PRIMARY_EXTS order, so a
    sidecar such as '<stem>.tif.aux.xml' or '<stem>.dbf' can never be returned.
    """
    for ext in PRIMARY_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _parse_epsg(crs: str) -> int | None:
    """'32634', 'EPSG:32634', 'epsg32634' -> 32634. Anything else -> None."""
    m = re.search(r"(\d{4,6})$", crs)
    return int(m.group(1)) if m else None


def _bbox_from_file(path: Path) -> list[float] | None:
    """Read the real extent from the file and convert it to WGS84."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            return [float(x) for x in transform_bounds(src.crs, "EPSG:4326", *src.bounds)]
    except Exception:
        pass
    try:  # vector fallback (GeoParquet, GPKG, GeoJSON, Shapefile...)
        import geopandas as gpd

        gdf = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
        # total_bounds is a numpy array: convert to plain floats, because the JSON
        # writer used by pystac (orjson) cannot serialize numpy.float64
        return [float(x) for x in gdf.to_crs(4326).total_bounds]
    except Exception:
        return None


def _bbox_from_name(lat: str, lon: str) -> list[float]:
    la, lo, h = float(lat), float(lon), FALLBACK_HALF_SIZE_DEG
    return [lo - h, la - h, lo + h, la + h]


def _checksum(path: Path) -> str:
    """Multihash sha2-256, as the STAC file extension expects."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "1220" + h.hexdigest()


# --------------------------------------------------------------------------
# Bucket helpers (the generic ones are in src/common.py)
# --------------------------------------------------------------------------
def _upload_run_files(run: RunAttributes, log=logger) -> None:
    """Upload the data files an item points to. The key is the path from 'datalake/'
    onwards, which mirrors the relative hrefs written into the items
    ('../../datalake/...' from catalog/<item>/<item>.json)."""
    files = [*run.gold_files.values(), run.embeddings_file, run.roads_file]
    for f in filter(None, files):
        prefix = bucket_key(f.parent)               # e.g. datalake/gold_zone
        if f.suffix.lower() == ".shp":
            keys = upload_shapefile_to_bucket(f, prefix, skip_if_same_size=True)
        else:
            keys = [upload_file_to_bucket(f, prefix, skip_if_same_size=True)]
        for k in keys:
            log.info("  in bucket: %s", k)


# --------------------------------------------------------------------------
# 1) Read attributes
# --------------------------------------------------------------------------
def read_run_attributes(
    gold_dir: Path = gold_zone_path,
    only_keys: set[str] | None = None,
) -> list[RunAttributes]:
    """Scan gold_zone and return one RunAttributes per (lat, lon, crs) run.

    only_keys: if given, keep only these run keys (e.g. {"39.88243_20.74780_EPSG4326"}).
    That avoids re-reading and re-hashing every old run each time the catalog is updated.
    """
    runs: dict[str, RunAttributes] = {}

    for path in sorted(p for p in gold_dir.iterdir() if _is_primary(p)):
        m = GOLD_PATTERN.match(path.stem)
        if not m:
            continue  # not one of our outputs
        lat, lon, crs, kind = m["lat"], m["lon"], m["crs"], m["kind"]
        key = f"{lat}_{lon}_{crs}"

        if only_keys is not None and key not in only_keys:
            continue

        if key not in runs:
            runs[key] = RunAttributes(
                lat=lat,
                lon=lon,
                crs=crs,
                epsg=_parse_epsg(crs),
                bbox=[],
                datetime_=EMBEDDINGS_SRC_DATE,
                created=_mtime(path),
                osm_snapshot=None,
            )
        run = runs[key]

        if kind in run.gold_files:
            logger.warning(f"  warning: {key} has several '{kind}' outputs; keeping "
                           f"{run.gold_files[kind].name}, ignoring {path.name}")
            continue
        run.gold_files[kind] = path
        run.created = max(run.created, _mtime(path))  # latest output wins

    for run in runs.values():
        # Companion inputs, matched by the same lat_lon_crs key
        run.embeddings_file = _find(retrieved_embeddings_dir, f"tessera_embeddings_{run.key}")
        run.roads_file = _find(osm_ingested_surfaces_dir, f"roads_{run.key}")
        if run.roads_file:
            run.osm_snapshot = _mtime(run.roads_file)

        # bbox: real extent first (the name only holds a point), name as fallback
        bbox = None
        for candidate in [*run.gold_files.values(), run.embeddings_file]:
            if candidate and (bbox := _bbox_from_file(candidate)):
                break
        run.bbox = bbox or _bbox_from_name(run.lat, run.lon)

    return list(runs.values())


# --------------------------------------------------------------------------
# 2) Create one Item
# --------------------------------------------------------------------------
def _make_asset(path: Path, title: str, roles: list[str], extra: dict | None = None) -> pystac.Asset:
    media_type = {
        ".tif": pystac.MediaType.GEOTIFF,
        ".tiff": pystac.MediaType.GEOTIFF,
        ".parquet": "application/vnd.apache.parquet",
        ".gpkg": "application/geopackage+sqlite3",
        ".geojson": pystac.MediaType.GEOJSON,
    }.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]

    extra_fields = dict(extra or {})
    if COMPUTE_CHECKSUMS:
        extra_fields["file:checksum"] = _checksum(path)
    extra_fields["file:size"] = path.stat().st_size

    return pystac.Asset(
        href=str(path.resolve()),  # made relative later, once the layout is known
        title=title,
        media_type=media_type,
        roles=roles,
        extra_fields=extra_fields,
    )


# asset keys end with the file type: per_road_inference_shp, embeddings_tif, ...
_KEY_SUFFIX = {".shp": "shp", ".tif": "tif", ".tiff": "tif"}


def _asset_key(base: str, path: Path) -> str:
    suffix = _KEY_SUFFIX.get(Path(path).suffix.lower())
    return f"{base}_{suffix}" if suffix else base


SIDECAR_MEDIA_TYPES = {
    "dbf": "application/x-dbf",
    "shx": "application/octet-stream",
    "prj": "text/plain",
    "cpg": "text/plain",
}


def _add_shapefile_sidecars(item: pystac.Item, key: str, shp_path: Path, title: str, roles: list[str]) -> None:
    """A STAC asset href points to ONE file, so the other parts of a shapefile
    (.dbf, .shx, .prj, .cpg, ...) are added as their own assets, named '<key>_<ext>'.
    The .shp asset itself stays the main one: GIS tools find its siblings by name."""
    shp_path = Path(shp_path)
    for part in sorted(shp_path.parent.glob(f"{shp_path.stem}.*")):
        if part == shp_path or not part.is_file():
            continue
        ext = part.name[len(shp_path.stem) + 1:]          # "dbf", "shp.xml", ...
        asset = _make_asset(part, f"{title} (.{ext})", roles)
        asset.media_type = SIDECAR_MEDIA_TYPES.get(ext.lower(), asset.media_type)
        item.add_asset(f"{key}_{ext.replace('.', '_')}", asset)


def create_item(run: RunAttributes) -> pystac.Item:
    west, south, east, north = run.bbox
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south]
        ]],
    }

    item = pystac.Item(
        id=f"{PREFIX}_{run.key}_{METHOD}_v{PIPELINE_VERSION}",
        geometry=geometry,
        bbox=run.bbox,
        datetime=run.datetime_,  # what the data represents
        properties={
            "created": run.created.isoformat().replace("+00:00", "Z"),
            "processing:software": {PIPELINE_NAME: PIPELINE_VERSION},
            f"{PREFIX}:method": METHOD,
            f"{PREFIX}:embedding_model": EMBEDDING_MODEL,
            f"{PREFIX}:embeddings_source_date": run.datetime_.date().isoformat(),
        },
        stac_extensions=[PROCESSING_EXT, FILE_EXT],
    )

    if run.epsg:
        ProjectionExtension.ext(item, add_if_missing=True).epsg = run.epsg
    if run.osm_snapshot:  # optional: leave out entirely when OSM was not used
        item.properties[f"{PREFIX}:osm_snapshot"] = run.osm_snapshot.date().isoformat()

    # Assets
    titles = {"per_road": "Per-road inference", "per_road_pixel": "Per-pixel inference"}
    for kind, path in run.gold_files.items():
        base = f"{kind}_inference"
        # main asset ends in its file type (_shp, _tif); a shapefile's parts in _dbf, _shx, _prj, ...
        key = _asset_key(base, path)
        item.add_asset(key, _make_asset(path, titles[kind], ["data"]))
        if path.suffix.lower() == ".shp":
            _add_shapefile_sidecars(item, base, path, titles[kind], ["data"])
    if run.embeddings_file:
        item.add_asset(
            _asset_key("embeddings", run.embeddings_file),
            _make_asset(run.embeddings_file, "Embeddings used as input", ["data", "source"]),
        )
    if run.roads_file:
        roads_key = _asset_key("osm_roads", run.roads_file)
        item.add_asset(
            roads_key,
            _make_asset(
                run.roads_file,
                "OSM roads used as input (clipped to tile)",
                ["metadata", "source"],
                {f"{PREFIX}:osm_snapshot": run.osm_snapshot.date().isoformat()},
            ),
        )
        if run.roads_file.suffix.lower() == ".shp":
            _add_shapefile_sidecars(
                item, "osm_roads", run.roads_file,
                "OSM roads used as input (clipped to tile)", ["metadata", "source"],
            )
    return item


# --------------------------------------------------------------------------
# 3) Create the Collection (the root, i.e. collection.json) and save everything
# --------------------------------------------------------------------------
def create_collection() -> pystac.Collection:
    """Shared info written once. The extent is recalculated from the Items later."""
    return pystac.Collection(
        id=COLLECTION_ID,
        title="Embedding Based Surface Classification (EBSC)",
        description=(
            "Surface material predictions (per surface and per pixel) derived from satellite embeddings and applied on OpenStreetMap surfaces for prediction."
        ),
        license="CC-BY-4.0",  # <- set the license you really want; check ODbL for OSM-derived data
        providers=[
            pystac.Provider(name="Your Name", roles=[pystac.ProviderRole.PRODUCER]),
            pystac.Provider(
                name="University of Cambridge, Energy and Environment Group (TESSERA embeddings)",
                roles=[pystac.ProviderRole.LICENSOR],
                url="https://github.com/ucam-eo/tessera",
            ),
            pystac.Provider(
                name="OpenStreetMap contributors",
                roles=[pystac.ProviderRole.LICENSOR],
                url="https://www.openstreetmap.org/copyright",
            ),
        ],
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
    )


def build_catalog(runs: list[RunAttributes], validate: bool = False) -> pystac.Collection:
    """Old behaviour: build a brand-new catalog from `runs` and save it locally only.
    Kept for reference; the pipeline now uses update_catalog()."""
    collection = create_collection()
    for run in runs:
        collection.add_item(create_item(run))

    collection.update_extent_from_items()
    collection.normalize_hrefs(str(STAC_CATALOG_PATH))      # catalog/collection.json, catalog/<item>/<item>.json
    collection.make_all_asset_hrefs_relative()        # ../../datalake/... instead of absolute paths
    if validate:  # needs internet to fetch the extension schemas
        collection.validate_all()
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return collection


def update_catalog(
    runs: list[RunAttributes],
    validate: bool = False,
    use_bucket: bool = True,
    log=logger,
) -> pystac.Collection:
    """Add `runs` to the existing catalog instead of creating a new one.

    use_bucket=True : read the current catalog from the bucket first, and upload the
                      result (data files -> item JSONs -> collection.json) afterwards.
    use_bucket=False: purely local (load local collection.json if present, else create).

    The catalog is always written locally to STAC_CATALOG_PATH.
    """
    catalog_dir = Path(STAC_CATALOG_PATH)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    local_collection = catalog_dir / "collection.json"

    # 1) load the existing catalog (bucket first, then local copy, else new)
    if use_bucket:
        ensure_bucket()
    if use_bucket and key_exists(COLLECTION_KEY):
        n = download_prefix(CATALOG_PREFIX, catalog_dir)
        log.info("Loaded existing catalog from bucket (%d file(s) downloaded)", n)
        source = "bucket"
    elif local_collection.exists():
        log.info("No catalog in the bucket, loading the local one: %s", local_collection)
        source = "local"
    else:
        log.info("No existing catalog found, creating a new one")
        source = "new"

    if source == "new":
        collection = create_collection()
    else:
        collection = pystac.Collection.from_file(str(local_collection))
        # providers are defined in create_collection(): refresh them so an existing
        # catalog picks up changes (e.g. a newly added provider)
        collection.providers = create_collection().providers
        # Old items keep working through the re-layout below: resolve their hrefs first,
        # everything is made relative again at the end.
        collection.make_all_asset_hrefs_absolute()

    # 2) add the new items; an item with the same id is replaced (re-run = update)
    for run in runs:
        item = create_item(run)
        if collection.get_item(item.id) is not None:
            collection.remove_item(item.id)
            log.info("Replacing existing item %s", item.id)
        collection.add_item(item)

    collection.update_extent_from_items()
    collection.normalize_hrefs(str(catalog_dir))      # catalog/collection.json, catalog/<item>/<item>.json
    collection.make_all_asset_hrefs_relative()        # ../../datalake/... instead of absolute paths
    if validate:  # needs internet to fetch the extension schemas
        collection.validate_all()

    # 3) local write (always)
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    log.info("Catalog written locally to %s", local_collection)

    # 4) upload: data files first, then item JSONs, collection.json last
    if use_bucket:
        for run in runs:
            _upload_run_files(run, log)
        json_files = sorted(catalog_dir.rglob("*.json"), key=lambda p: p.name == "collection.json")
        for p in json_files:
            key = f"{CATALOG_PREFIX}/{p.relative_to(catalog_dir).as_posix()}"
            upload_file(p, key)   # JSON must always overwrite
        log.info("Catalog uploaded to s3://%s/%s", BUCKET, COLLECTION_KEY)

    return collection


# --------------------------------------------------------------------------