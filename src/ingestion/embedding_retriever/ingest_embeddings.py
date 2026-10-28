from pathlib import Path
from typing import Optional, Sequence
import inspect
import os

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from geotessera import GeoTessera


datalake_raw_zone_path = Path(
    "E:/Programming/SIG/surface_type_feasibility_analysis/"
    "tessera_approach/datalake/raw_zone"
)


def generate_tessera_embeddings(
    aoi_shapefile: str,
    output_dir: str,
    year: int = 2024,
    dataset_version: str = "v1",
    dataset_variant: Optional[str] = None,
    bands: Optional[Sequence[int]] = None,
    clip_to_aoi: bool = True,
    mosaic: bool = True,
    cache_dir: Optional[str] = None,
    logger=None,
) -> dict:
    """Generate TESSERA embeddings covering an AOI given as a shapefile."""

    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(
        "Starting TESSERA embedding generation | AOI=%s | year=%s | "
        "dataset_version=%s | dataset_variant=%s",
        aoi_shapefile,
        year,
        dataset_version,
        dataset_variant,
    )

    try:
        # --- 1. Prepare output directory --------------------------------
        output_dir = Path(output_dir)

        logger.info("Preparing output directory: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        aoi_shapefile_base_name = os.path.basename(aoi_shapefile)
        file_name_without_ext, _ = os.path.splitext(
            aoi_shapefile_base_name
        )

        output_path = (
            output_dir
            / f"{file_name_without_ext.replace('aoi', 'tessera_embeddings')}.tif"
        )

        logger.info("Embedding mosaic output path: %s", output_path)

        # --- 2. Load AOI and ensure WGS84 -------------------------------
        logger.info("Reading AOI shapefile: %s", aoi_shapefile)
        aoi = gpd.read_file(aoi_shapefile)

        logger.info(
            "AOI loaded | features=%d | CRS=%s",
            len(aoi),
            aoi.crs,
        )

        if aoi.empty:
            raise ValueError(f"No features found in {aoi_shapefile}")

        if aoi.crs is None:
            raise ValueError(
                "AOI shapefile has no CRS defined; cannot confirm it's WGS84."
            )

        if aoi.crs.to_epsg() != 4326:
            logger.info("Reprojecting AOI from %s to EPSG:4326", aoi.crs)
            aoi = aoi.to_crs(epsg=4326)
            logger.info("AOI reprojection completed")

        logger.info("Dissolving AOI geometries")
        aoi_geom = (
            aoi.geometry.union_all()
            if hasattr(aoi.geometry, "union_all")
            else aoi.unary_union
        )

        min_lon, min_lat, max_lon, max_lat = aoi_geom.bounds
        bbox = (min_lon, min_lat, max_lon, max_lat)

        logger.info("AOI bounding box: %s", bbox)

        # --- 3. Initialize GeoTessera ----------------------------------
        logger.info("Inspecting GeoTessera constructor parameters")

        gt_init_params = inspect.signature(
            GeoTessera.__init__
        ).parameters

        gt_kwargs = {
            "dataset_version": dataset_version,
            "cache_dir": cache_dir,
        }

        if "dataset_variant" in gt_init_params:
            gt_kwargs["dataset_variant"] = dataset_variant
        elif dataset_variant is not None:
            raise TypeError(
                "Installed geotessera version does not support "
                "'dataset_variant'."
            )

        logger.info(
            "Initializing GeoTessera | version=%s | cache_dir=%s",
            dataset_version,
            cache_dir,
        )

        gt = GeoTessera(**gt_kwargs)

        logger.info("GeoTessera initialization completed")

        # --- 4. Query embedding tiles ----------------------------------
        logger.info(
            "Querying TESSERA registry | bbox=%s | year=%s",
            bbox,
            year,
        )

        tiles_to_fetch = gt.registry.load_blocks_for_region(
            bounds=bbox,
            year=year,
        )

        logger.info(
            "TESSERA registry query completed | tiles_found=%d",
            len(tiles_to_fetch) if tiles_to_fetch else 0,
        )

        if not tiles_to_fetch:
            raise ValueError(
                f"No TESSERA tiles found for bbox {bbox}, year={year}, "
                f"dataset_version={dataset_version}."
            )

        # --- 5. Download/export embedding tiles ------------------------
        logger.info(
            "Starting embedding tile export/download | tile_count=%d | "
            "output_dir=%s | bands=%s",
            len(tiles_to_fetch),
            output_dir,
            list(bands) if bands is not None else "all",
        )

        tile_files = gt.export_embedding_geotiffs(
            tiles_to_fetch=tiles_to_fetch,
            output_dir=str(output_dir),
            bands=list(bands) if bands is not None else None,
            compress="lzw",
        )

        logger.info(
            "Embedding tile export completed | files_returned=%d",
            len(tile_files) if tile_files else 0,
        )

        for tile_file in tile_files:
            logger.info("Embedding tile available: %s", tile_file)

        result = {
            "tile_files": tile_files,
            "mosaic_file": None,
            "bbox": bbox,
            "n_tiles": len(tile_files),
        }

        # --- 6. Mosaic and optionally clip ------------------------------
        if mosaic and tile_files:
            logger.info(
                "Starting mosaic creation | tile_count=%d",
                len(tile_files),
            )

            srcs = []

            try:
                for tile_file in tile_files:
                    logger.info("Opening raster tile: %s", tile_file)
                    srcs.append(rasterio.open(tile_file))

                logger.info("Merging raster tiles into mosaic")
                merged_array, merged_transform = merge(srcs)

                merged_crs = srcs[0].crs

                logger.info(
                    "Mosaic merge completed | shape=%s | CRS=%s",
                    merged_array.shape,
                    merged_crs,
                )

                profile = srcs[0].profile.copy()
                profile.update(
                    height=merged_array.shape[1],
                    width=merged_array.shape[2],
                    transform=merged_transform,
                    count=merged_array.shape[0],
                    BIGTIFF="YES",
                )

                logger.info("Writing mosaic to: %s", output_path)

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(merged_array)

                logger.info("Mosaic written successfully")

                if clip_to_aoi:
                    logger.info("Reprojecting AOI geometry to mosaic CRS")

                    aoi_in_mosaic_crs = gpd.GeoSeries(
                        [aoi_geom],
                        crs="EPSG:4326",
                    ).to_crs(merged_crs)

                    logger.info("Clipping mosaic to AOI geometry")

                    with rasterio.open(output_path) as src:
                        clipped_array, clipped_transform = rio_mask(
                            src,
                            aoi_in_mosaic_crs.geometry,
                            crop=True,
                        )

                        clipped_profile = src.profile.copy()
                        clipped_profile.update(
                            height=clipped_array.shape[1],
                            width=clipped_array.shape[2],
                            transform=clipped_transform,
                        )

                    logger.info(
                        "AOI clipping completed | clipped_shape=%s",
                        clipped_array.shape,
                    )

                    logger.info("Writing clipped mosaic: %s", output_path)

                    with rasterio.open(
                        output_path, "w", **clipped_profile
                    ) as dst:
                        dst.write(clipped_array)

                    result["mosaic_file"] = str(output_path)

                    logger.info("Clipped mosaic written successfully")
                else:
                    result["mosaic_file"] = str(output_path)

                logger.info("Mosaic processing completed")

            finally:
                for src in srcs:
                    src.close()

        elif not mosaic:
            logger.info("Mosaic disabled; retaining individual tile files")
        else:
            logger.info("No tile files returned; skipping mosaic creation")

        logger.info(
            "TESSERA embedding generation completed successfully | "
            "tiles=%d | mosaic=%s",
            result["n_tiles"],
            result["mosaic_file"],
        )

        return result

    except Exception:
        logger.exception("TESSERA embedding generation failed")
        raise