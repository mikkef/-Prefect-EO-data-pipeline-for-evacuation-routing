"""
inference.py

Use a trained model to classify the surface of roads.

  load_saved_model()              load best_model.pkl written by train_best_model()
  infer_roads()                   one class per road (majority of its pixels)
  create_pixel_inference_raster() one class + paved/unpaved probability per pixel
  run_inference()                 runs both for one raster / roads shapefile

The model only needs predict(), predict_proba() and classes_, so it can be
either the in-memory model returned by train_best_model() or the one loaded
from disk with load_saved_model().
"""

from pathlib import Path
from typing import Optional, Sequence
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


# ---------------------------------------------------------------------------
# 0. Load a saved model
# ---------------------------------------------------------------------------
class SavedModel:
    """Model loaded from disk: predict() returns the original class names."""

    def __init__(self, pipeline, classes):
        self.pipeline = pipeline
        self.classes_ = np.array(classes)

    def predict(self, X):
        return self.classes_[self.pipeline.predict(X)]

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def load_saved_model(model_path: str, logger):
    """
    Load a model saved by train_best_model().

    Returns
    -------
    model             object with predict(), predict_proba(), classes_
    selected_features list of feature names, in the order used for training
    """
    bundle = joblib.load(model_path)

    model = SavedModel(bundle["pipeline"], bundle["classes"])

    logger.info(f"Loaded model: {bundle.get('model_name')}")
    logger.info(f"  Features: {len(bundle['features'])}")
    logger.info(f"  Classes:  {bundle['classes']}")

    return model, list(bundle["features"])


# ---------------------------------------------------------------------------
# Helpers shared by both inference functions
# ---------------------------------------------------------------------------
def _resolve_features(src, selected_features, band_prefix):
    """Band names of the raster and the indices of the features the model uses."""
    band_names = [f"{band_prefix}{i}" for i in range(src.count)]

    if selected_features is None:
        feature_indices = list(range(src.count))
        feature_names = band_names
    else:
        missing_features = [f for f in selected_features if f not in band_names]

        if missing_features:
            raise ValueError(
                "Selected features are not present in raster: "
                f"{missing_features}"
            )

        feature_indices = [band_names.index(f) for f in selected_features]
        feature_names = list(selected_features)

    return feature_indices, feature_names


def _road_pixels(geometry, buffer_distance, shape, transform, data, nodata):
    """Row/col of the valid raster pixels covered by one road."""
    if buffer_distance > 0:
        geometry = geometry.buffer(buffer_distance)
        all_touched = False
    else:
        all_touched = True

    road_mask = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)

    if nodata is not None:
        valid = road_mask & np.all(data != nodata, axis=0)
    else:
        valid = road_mask & np.all(np.isfinite(data), axis=0)

    return np.where(valid)


def _read_roads(roads_path):
    roads = gpd.read_file(roads_path)

    if roads.empty:
        raise ValueError(f"No features found in {roads_path}")

    if roads.crs is None:
        raise ValueError(f"{roads_path} has no CRS defined.")

    return roads


# ---------------------------------------------------------------------------
# 1. Infer surface class for each road
# ---------------------------------------------------------------------------
def infer_roads(
    logger,
    raster_path: str,
    roads_path: str,
    model,
    output_path: str = "surface_inferred_roads.shp",
    buffer_distance: float = 5.0,
    threshold: float = 0.60,
    band_prefix: str = "B",
    selected_features: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Every road gets the class most of its pixels were predicted as, if that
    share ("confidence") is at least `threshold`; otherwise "uncertain".
    """
    roads = _read_roads(roads_path)

    with rasterio.open(raster_path) as src:

        roads = roads.to_crs(src.crs)

        transform = src.transform
        shape = (src.height, src.width)
        nodata = src.nodata

        data = src.read()

        feature_indices, feature_names = _resolve_features(
            src, selected_features, band_prefix
        )

        logger.info("\nInference features:")
        logger.info(f"Using {len(feature_names)} features:")
        logger.info(feature_names)

        roads["inference"] = "uncertain"
        roads["confidence"] = np.nan
        roads["n_pixels"] = 0

        selected_data = data[feature_indices, :, :]

        for idx, road in roads.iterrows():

            geometry = road.geometry

            if geometry is None or geometry.is_empty:
                continue

            rows, cols = _road_pixels(
                geometry, buffer_distance, shape, transform, data, nodata
            )

            roads.at[idx, "n_pixels"] = len(rows)

            if len(rows) == 0:
                continue

            # Extract ONLY the model's features
            X = selected_data[:, rows, cols].T.astype("float64")

            finite = np.all(np.isfinite(X), axis=1)
            X = X[finite]

            if len(X) == 0:
                roads.at[idx, "n_pixels"] = 0
                continue

            # Scaling (if any) runs inside the model pipeline
            predictions = model.predict(X)

            counts = pd.Series(predictions).value_counts()

            dominant_class = counts.index[0]
            confidence = counts.iloc[0] / len(predictions)

            roads.at[idx, "confidence"] = confidence
            roads.at[idx, "n_pixels"] = len(predictions)

            if confidence >= threshold:
                roads.at[idx, "inference"] = str(dominant_class)
            else:
                roads.at[idx, "inference"] = "uncertain"

            logger.info(
                f"Road {idx}: {roads.at[idx, 'inference']} "
                f"({confidence:.1%}, {len(predictions)} pixels)"
            )

    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    roads.to_file(output_path, driver="ESRI Shapefile")

    logger.info(f"\nWrote inferred roads -> {output_path}")
    logger.info("\nInference summary:")
    logger.info(roads["inference"].value_counts())

    return roads


# ---------------------------------------------------------------------------
# 2. Pixel-level inference raster
# ---------------------------------------------------------------------------
UNPAVED_NAMES = {
    "1",
    "unpaved",
    "compact",
    "compacted",
    "compacted_earth",
    "dirt",
    "gravel",
    "soil",
    "earth",
    "grass",
    "mud",
}

PAVED_NAMES = {
    "0",
    "paved",
    "asphalt",
    "concrete",
    "paving_stones",
    "paving stones",
    "cobblestone",
    "cobblestones",
    "brick",
}


def create_pixel_inference_raster(
    logger,
    raster_path: str,
    roads_path: str,
    model,
    output_path: str = "pixel_inference.tif",
    buffer_distance: float = 5.0,
    band_prefix: str = "B",
    selected_features: Optional[Sequence[str]] = None,
) -> dict:
    """
    GeoTIFF with 3 bands:
        1  predicted class code (see the `class_encoding` tag)
        2  paved probability
        3  unpaved probability
    """
    roads = _read_roads(roads_path)

    with rasterio.open(raster_path) as src:

        roads = roads.to_crs(src.crs)

        transform = src.transform
        height = src.height
        width = src.width
        nodata = src.nodata

        data = src.read()

        feature_indices, feature_names = _resolve_features(
            src, selected_features, band_prefix
        )

        logger.info("\nPixel inference features:")
        logger.info(f"Using {len(feature_names)} features:")
        logger.info(feature_names)

        # ------------------------------------------------------------
        # Model classes
        # ------------------------------------------------------------
        model_classes = [str(c).strip().lower() for c in model.classes_]

        class_to_code = {name: i + 1 for i, name in enumerate(model_classes)}

        paved_indices = []
        unpaved_indices = []

        for i, class_name in enumerate(model_classes):

            if class_name in UNPAVED_NAMES:
                unpaved_indices.append(i)

            elif class_name in PAVED_NAMES:
                paved_indices.append(i)

            else:
                raise ValueError(
                    f"Unknown surface class '{class_name}'. "
                    "Add it to PAVED_NAMES or UNPAVED_NAMES."
                )

        logger.info("\nModel classes:")
        logger.info(model_classes)
        logger.info("\nPaved classes:")
        logger.info([model_classes[i] for i in paved_indices])
        logger.info("\nUnpaved classes:")
        logger.info([model_classes[i] for i in unpaved_indices])

        # ------------------------------------------------------------
        # Output arrays
        # ------------------------------------------------------------
        pixel_class = np.zeros((height, width), dtype="uint16")
        paved_confidence = np.zeros((height, width), dtype="float32")
        unpaved_confidence = np.zeros((height, width), dtype="float32")

        selected_data = data[feature_indices, :, :]

        # ------------------------------------------------------------
        # Process roads
        # ------------------------------------------------------------
        for idx, road in roads.iterrows():

            geometry = road.geometry

            if geometry is None or geometry.is_empty:
                continue

            rows, cols = _road_pixels(
                geometry, buffer_distance, (height, width), transform, data, nodata
            )

            if len(rows) == 0:
                continue

            X = selected_data[:, rows, cols].T.astype("float64")

            finite = np.all(np.isfinite(X), axis=1)

            if not np.any(finite):
                continue

            valid_rows = rows[finite]
            valid_cols = cols[finite]
            X = X[finite]

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            predicted_codes = np.array(
                [class_to_code[str(p).strip().lower()] for p in predictions],
                dtype="uint16",
            )

            if paved_indices:
                paved_prob = probabilities[:, paved_indices].sum(axis=1)
            else:
                paved_prob = np.zeros(len(X), dtype="float32")

            if unpaved_indices:
                unpaved_prob = probabilities[:, unpaved_indices].sum(axis=1)
            else:
                unpaved_prob = np.zeros(len(X), dtype="float32")

            pixel_class[valid_rows, valid_cols] = predicted_codes
            paved_confidence[valid_rows, valid_cols] = paved_prob
            unpaved_confidence[valid_rows, valid_cols] = unpaved_prob

            logger.info(f"Processed road {idx}: {len(X)} pixels")

        # ------------------------------------------------------------
        # Write GeoTIFF
        # ------------------------------------------------------------
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=3,
            dtype="float32",
            compress="deflate",
            nodata=0.0,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:

            dst.write(pixel_class.astype("float32"), 1)
            dst.write(paved_confidence, 2)
            dst.write(unpaved_confidence, 3)

            dst.set_band_description(1, "Predicted pixel class")
            dst.set_band_description(2, "Paved probability")
            dst.set_band_description(3, "Unpaved probability")

            dst.update_tags(
                class_encoding=str(class_to_code),
                paved_classes=",".join(model_classes[i] for i in paved_indices),
                unpaved_classes=",".join(model_classes[i] for i in unpaved_indices),
                selected_features=",".join(feature_names),
            )

    logger.info(f"\nWrote pixel inference raster -> {output_path}")

    return {
        "output_path": output_path,
        "class_encoding": class_to_code,
        "paved_classes": [model_classes[i] for i in paved_indices],
        "unpaved_classes": [model_classes[i] for i in unpaved_indices],
        "selected_features": feature_names,
    }


# ---------------------------------------------------------------------------
# 3. Run both for one raster / roads shapefile
# ---------------------------------------------------------------------------
def infer_raster(
    logger,
    raster_path: str,
    roads_path: str,
    model=None,
    model_path: Optional[str] = None,
    selected_features: Optional[Sequence[str]] = None,
    road_output_path: str = "surface_inferred_roads.shp",
    pixel_output_path: str = "surface_roads_per_pixel_inference.tif",
    buffer_distance: float = 3.0,
    threshold: float = 0.70,

) -> dict:
    """
    Pass either a trained `model` (+ `selected_features`) or the path of a
    saved model (`model_path`); with a path, the features stored with the
    model are used.
    """
    if model is None:
        if model_path is None:
            raise ValueError("Pass either `model` or `model_path`.")
        model, saved_features = load_saved_model(model_path, logger)
        if selected_features is None:
            selected_features = saved_features

    # ------------------------------------------------------------
    # BEST MODEL -> ROAD INFERENCE
    # ------------------------------------------------------------
    inferred_roads = infer_roads(
        raster_path=raster_path,
        roads_path=roads_path,
        model=model,
        output_path=road_output_path,
        buffer_distance=buffer_distance,
        threshold=threshold,
        selected_features=selected_features,
        logger=logger
    )

    # ------------------------------------------------------------
    # BEST MODEL -> PIXEL INFERENCE
    # ------------------------------------------------------------
    pixel_raster_info = create_pixel_inference_raster(
        raster_path=raster_path,
        roads_path=roads_path,
        model=model,
        output_path=pixel_output_path,
        buffer_distance=buffer_distance,
        selected_features=selected_features,
        logger=logger
    )

    return {
        # "inferred_roads": inferred_roads,
        "pixel_raster_info": pixel_raster_info,
    }