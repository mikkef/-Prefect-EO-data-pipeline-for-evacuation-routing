"""
sample_gatherer_utils.py

Build the labeled pixel table used to train the surface classifier.

  mask_pixels_to_roads()        keep only raster pixels that intersect roads
  collect_training_data()       one raster + one road shapefile -> labeled pixels
  collect_all_training_data()   loop over every embeddings raster and its
                                matching roads shapefile, concatenate the result

Every row of the table is one pixel:
    x, y, B0 ... B127, class_label, road_id
"""

from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from rasterio.features import rasterize


# ---------------------------------------------------------------------------
# 1. Mask: keep only pixels intersecting the road network
# ---------------------------------------------------------------------------
def mask_pixels_to_roads(
    raster_path: str,
    roads_path: str,
    buffer_distance: float = 0.0,
    band_prefix: str = "B",
    output_raster_path: Optional[str] = None,
    return_dataframe: bool = True,
) -> Optional[pd.DataFrame]:

    with rasterio.open(raster_path) as src:

        raster_crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width
        nodata = src.nodata
        profile = src.profile.copy()

        roads = gpd.read_file(roads_path)

        if roads.empty:
            raise ValueError(f"No features found in {roads_path}")

        if roads.crs is None:
            raise ValueError(f"{roads_path} has no CRS defined.")

        roads = roads.to_crs(raster_crs)

        if buffer_distance and buffer_distance > 0:
            geoms = roads.geometry.buffer(buffer_distance)
            all_touched = False
        else:
            geoms = roads.geometry
            all_touched = True

        road_mask = rasterize(
            [(geom, 1) for geom in geoms],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)

        band1 = src.read(1)

        if nodata is not None:
            valid = road_mask & (band1 != nodata)
        else:
            valid = road_mask & ~np.isnan(band1)

        rows, cols = np.where(valid)

        if len(rows) == 0:
            raise ValueError("No raster pixels intersect the road network.")

        data = src.read()

        if output_raster_path is not None:

            out_nodata = nodata if nodata is not None else 0

            masked_data = np.where(valid[np.newaxis, :, :], data, out_nodata)

            out_profile = profile.copy()
            out_profile.update(nodata=out_nodata)

            Path(output_raster_path).parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(output_raster_path, "w", **out_profile) as dst:
                dst.write(masked_data.astype(profile["dtype"]))

            print(f"Wrote masked raster -> {output_raster_path}")

        if not return_dataframe:
            return None

        values = data[:, rows, cols].T

        xs, ys = rasterio.transform.xy(transform, rows, cols)

    band_cols = [f"{band_prefix}{i}" for i in range(data.shape[0])]

    df = pd.DataFrame(values, columns=band_cols)
    df.insert(0, "y", ys)
    df.insert(0, "x", xs)

    print(f"Kept {len(df)} / {height * width} pixels intersecting roads.")

    return df


# ---------------------------------------------------------------------------
# 2. Training data: assign class and road id per pixel
# ---------------------------------------------------------------------------
def collect_training_data(
    raster_path: str,
    roads_path: str,
    label_field: str = "label",
    class_field: str = "category_1",
    buffer_distance: float = 0.0,
    band_prefix: str = "B",
    allowed_classes: Sequence[str] = ("paved", "unpaved"),
) -> pd.DataFrame:
    """
    Labeled road pixels of ONE raster.

    Only roads where `label_field` is true and whose `class_field` value is
    in `allowed_classes` are used. Each pixel gets the class and the
    feature index (road_id) of the road it falls on.
    """
    roads = gpd.read_file(roads_path)

    if roads.empty:
        raise ValueError(f"No features found in {roads_path}")

    for col in (label_field, class_field):
        if col not in roads.columns:
            raise ValueError(f"Column '{col}' not found in {roads_path}")

    def _is_true(v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", 1)
        return bool(v)

    roads_labeled = roads[
        roads[label_field].apply(_is_true)
        & roads[class_field]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([c.lower() for c in allowed_classes])
    ].copy()

    if roads_labeled.empty:
        raise ValueError(f"No roads with {label_field} == True found.")

    # Class name of each road feature (same order as roads_labeled)
    classes = roads_labeled[class_field].astype(str).to_numpy()
    unique_classes = sorted(set(classes))

    # Original feature index of each road (this becomes road_id)
    road_index = roads_labeled.index.to_numpy()

    with rasterio.open(raster_path) as src:

        raster_crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width
        nodata = src.nodata

        if roads_labeled.crs is None:
            raise ValueError(f"{roads_path} has no CRS.")

        roads_labeled = roads_labeled.to_crs(raster_crs)

        if buffer_distance and buffer_distance > 0:
            geoms = roads_labeled.geometry.buffer(buffer_distance)
            all_touched = False
        else:
            geoms = roads_labeled.geometry
            all_touched = True

        # Burn a unique number per road feature (1..N; 0 = no road).
        # The class is looked up from the road afterwards.
        feature_numbers = np.arange(1, len(roads_labeled) + 1)

        burned = rasterize(
            list(zip(geoms, feature_numbers)),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="int32",
        )

        band1 = src.read(1)

        pixel_mask = burned > 0

        if nodata is not None:
            valid = pixel_mask & (band1 != nodata)
        else:
            valid = pixel_mask & ~np.isnan(band1)

        rows, cols = np.where(valid)

        if len(rows) == 0:
            raise ValueError("No raster pixels intersect the labeled roads.")

        data = src.read()

        values = data[:, rows, cols].T

        xs, ys = rasterio.transform.xy(transform, rows, cols)

        pixel_feature = burned[rows, cols]  # 1..N, which road each pixel is on

    band_cols = [f"{band_prefix}{i}" for i in range(data.shape[0])]

    df = pd.DataFrame(values, columns=band_cols)

    df.insert(0, "y", ys)
    df.insert(0, "x", xs)

    df["class_label"] = classes[pixel_feature - 1]
    df["road_id"] = road_index[pixel_feature - 1]

    print(
        f"Collected {len(df)} labeled pixels across "
        f"{len(unique_classes)} classes and "
        f"{df['road_id'].nunique()} roads:"
    )

    print(df["class_label"].value_counts())

    return df


# ---------------------------------------------------------------------------
# 3. Training data for every AOI
# ---------------------------------------------------------------------------
def collect_all_training_data(
    embeddings_dir,
    roads_dir,
    label_field: str = "LABELING",
    class_field: str = "surface_L1",
    buffer_distance: float = 3.0,
    allowed_classes: Sequence[str] = ("paved", "unpaved"),
    embeddings_prefix: str = "tessera_embeddings_",
    roads_prefix: str = "roads_",
    roads_suffix: str = "_02_edit.shp",
    unique_road_ids: bool = True,
) -> pd.DataFrame:
    """
    For every embeddings raster

        <embeddings_dir>/tessera_embeddings_<aoi>.tif

    look for its roads shapefile

        <roads_dir>/roads_<aoi>_02_edit.shp

    collect the labeled pixels, and concatenate everything.

    unique_road_ids
        Each shapefile numbers its roads from 0, so "road 5" of one AOI and
        "road 5" of another AOI would share the same road_id after the
        tables are concatenated. With True the AOI name is added to the id
        ("<aoi>_5"), so different roads never share an id.
    """
    embeddings_dir = Path(embeddings_dir)
    roads_dir = Path(roads_dir)

    training_dfs = []

    for raster in sorted(embeddings_dir.glob("*.tif")):
        # Example:
        # tessera_embeddings_39.56223_21.71406_EPSG4326.tif

        # Remove "tessera_embeddings_" prefix
        suffix = raster.stem.replace(embeddings_prefix, "")

        # Construct roads filename
        roads_file_path = roads_dir / f"{roads_prefix}{suffix}{roads_suffix}"

        print(f"##### {roads_file_path}")

        if roads_file_path.exists():
            print(f"Raster: {raster.name}")
            print(f"AOI: {roads_file_path}")

            training_df = collect_training_data(
                raster_path=str(raster),
                roads_path=str(roads_file_path),
                label_field=label_field,
                class_field=class_field,
                buffer_distance=buffer_distance,
                allowed_classes=allowed_classes,
            )

            if unique_road_ids:
                training_df["road_id"] = (
                    suffix + "_" + training_df["road_id"].astype(str)
                )

            training_dfs.append(training_df)
        else:
            print(f"WARNING road shapefile not found for: {raster.name}")

    if not training_dfs:
        raise ValueError("No training data was collected.")

    return pd.concat(training_dfs, ignore_index=True)
