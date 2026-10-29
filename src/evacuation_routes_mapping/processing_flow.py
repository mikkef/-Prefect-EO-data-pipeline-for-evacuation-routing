import geopandas as gpd
import subprocess
import os
from pathlib import Path
from shapely.geometry import Point
from prefect import flow, get_run_logger
from constants import inference_aois_dir, osm_ingested_surfaces_dir, silver_zone_path


def get_logger():
    return get_run_logger()


@flow(name="Evacuation Routes Flow", log_prints=True)
async def evacuation_routes_processing():
    """
    Handles the evacuation routes subproduct analysis.
    Create the evacuation_routes_point.shp and evacuation_routes_line.shp files.
    """
    logger = get_logger()

    # find inference aoi shp
    aoi_shp_file_path = next(Path(inference_aois_dir).glob("*.shp"), None)
    if aoi_shp_file_path is None:
        raise FileNotFoundError(
            f"No .shp file found in {inference_aois_dir}"
        )

    # calculate evacuation_routes_start_point
    evacuation_route_starting_point_shp_file_path =  await create_evacuation_routes_start_point(aoi_shp_file_path)

    aoi_gdf = gpd.read_file(aoi_shp_file_path)

    # TODO ADD THIS TO A PREPROCESS STEP
    #######################################
    aoi_gdf['AOI'] = "AOI01"
    aoi_gdf.to_file(aoi_shp_file_path)
    ########################################

    evacuation_route_starting_point_gdf = gpd.read_file(evacuation_route_starting_point_shp_file_path)
    # Reproject to aoi crs
    evacuation_route_starting_point_gdf = evacuation_route_starting_point_gdf.to_crs(aoi_gdf.crs)
    # Extract the first (and should be only) geometry
    point_geom = evacuation_route_starting_point_gdf.geometry.iloc[0]
    # Get coordinates as [x, y]
    # Format as 'x,y' string
    coordinates = f"{point_geom.x},{point_geom.y}"
    logger.debug(f"Evacuation routes start projected coordinates: {coordinates}")

    await area_based_processing(aoi_shp_file_path, coordinates)


@flow(name="Evacuation Routes Calculation and Generation", log_prints=True)
async def area_based_processing(aoi_shp_file_path, coordinates: str):
    """
    Handles the area based processing module block.
    :param src_dir: The pythonpath.
    :param coordinates: The coordinates for calculating the evacuation routes.
    :return:
    """
    # Define required paths
    logger = get_logger()
    logger.info("Evacuation Routes Calculation and Generation subflow started...")
    current_path = os.path.dirname(os.path.abspath(__file__))
    relative_script_path_er = os.path.join("scripts", "app.py")
    script_er = os.path.join(current_path, relative_script_path_er)
    output_path_er = silver_zone_path

    # evacuation app.py args
    evac_args = ["-a",
                 str(aoi_shp_file_path),
                 "-s",
                 coordinates,  # '42.6263248,13.2925175',
                 "-o",
                 str(output_path_er)
                 ]

    # run evacuation app.py
    result = subprocess.run(["python", script_er] + evac_args, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(
            "Evacuation Routes Calculation and Generation-> Evacuation Routes Calculation and Generation subflow finished successfully")
    else:
        logger.error(
            f"Evacuation Routes Calculation and Generation-> Evacuation Routes Calculation and Generation subflow failed with return code: {result.returncode}")
        raise RuntimeError(f"[S8-02-MG03-MB02] Area-based processing -> Error output: {result.stderr}")

    logger.info("Evacuation Routes Calculation and Generation subflow finished...")


@flow(name="Evacuation Routes Start Point Calculation", log_prints=True)
async def create_evacuation_routes_start_point(aoi_shp_file_path):
    logger = get_run_logger()
    # Output evacuation route starting point shapefile
    evacuation_route_starting_point_shp_file_path = (
            osm_ingested_surfaces_dir / "evacuation_routes_start.shp"
    )

    # Read AOI shapefile
    aoi_gdf = gpd.read_file(aoi_shp_file_path)

    # Get bounding box: minx, miny, maxx, maxy
    logger.info("Calculating AOI bounding box center..")
    minx, miny, maxx, maxy = aoi_gdf.total_bounds

    # Calculate the center of the bounding box
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    logger.info(f"BBOX center: {center_x}, {center_y}..")

    # Create a point feature
    logger.info("Creating shapefile..")
    center_point = Point(center_x, center_y)

    # Create a GeoDataFrame using the AOI's CRS
    start_point_gdf = gpd.GeoDataFrame(
        {"id": [1], "name": ["evacuation_routes_start"]},
        geometry=[center_point],
        crs=aoi_gdf.crs,
    )

    # Save as a shapefile
    start_point_gdf.to_file(
        evacuation_route_starting_point_shp_file_path,
        driver="ESRI Shapefile",
    )

    if not evacuation_route_starting_point_shp_file_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {evacuation_route_starting_point_shp_file_path}"
        )

    logger.info(f"Evacuation starting point saved to: "
          f"{evacuation_route_starting_point_shp_file_path}")
    logger.info(f"Coordinates: ({center_x}, {center_y})")

    return evacuation_route_starting_point_shp_file_path