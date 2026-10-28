import os
import click
import pandas as pd
from loguru import logger
from src.evacuation_routes_mapping.scripts.evacuation_routes import AOIRoutePlanner
import geopandas as gpd
from shapely.geometry import Point

# aoi_path = r"C:\NoSync\IRIDE\DevOps_demo\OU-S8-02-04\Evacuation routes\Use cases\UC 1\Input\AoI.shp"
# start_point = '39.1348286,26.5425057'
# outfolder = r"C:\NoSync\IRIDE\DevOps_demo\OU-S8-02-04\Evacuation routes\Use cases\UC 1\Output"
#

INT32_OUTPUT_COLUMNS = [
    'Graph_Rot',
    'ID',
    'SQKM',
    'lanes',
    'maxspeed',
    'osmid',
    'id'
]



@click.command()
@click.option('-a', 'aoi_path',
              default=None,
              type=click.Path(resolve_path=True),
              required=True,
              help='Path to AOI dataset. Must be a polygon vector dataset with 1 feature')
@click.option('-s','start_point',
              required=True,
              help='Coordinates for start point, in Χ,Υ format in the same CRS as AOI')
@click.option('-o', 'outfolder',
              required=True,
              default=None,
              type=click.Path(resolve_path=True),
              help='Folder path for Output')

def run_evacuation(aoi_path,start_point,outfolder):
    # Check AOI path
    if not os.path.exists(aoi_path):
        logger.error(f"AOI path does not exist: {aoi_path}")
        raise click.BadParameter("AOI path does not exist.")

    try:
        aoi = gpd.read_file(aoi_path)
    except Exception as e:
        logger.error(f"Failed to read AOI file: {e}")
        raise click.BadParameter("Cannot read AOI dataset.")

    if aoi.geom_type.unique()[0] != 'Polygon' and aoi.geom_type.unique()[0] != 'MultiPolygon':
        logger.error("AOI dataset must be a polygon vector dataset.")
        raise click.BadParameter("AOI dataset must be a polygon vector dataset.")

    if len(aoi) != 1:
        logger.error("AOI dataset must contain exactly 1 feature.")
        raise click.BadParameter("AOI dataset must contain exactly 1 feature.")

    # Check start_point format
    try:
        lat, lon = map(float, start_point.split(','))
    except Exception:
        logger.error("Start point must be in 'X,Y' (without text brackets) format with numeric values.")
        raise click.BadParameter("Start point must be in 'X,Y' (without text brackets) format with numeric values.")

    # if not (-90 <= lat <= 90 and -180 <= lon <= 180):
    #     logger.warning("Start point coordinates are out of valid range.")
    #     raise click.BadParameter("Start point coordinates are out of valid range.")

    # Check output folder
    if not os.path.isdir(outfolder):
        logger.error(f"Output folder does not exist: {outfolder}")
        raise click.BadParameter("Output folder does not exist.")

    logger.info("All input parameters validated successfully.")
    X = start_point.split(',')[0]
    Y = start_point.split(',')[1]
    point = gpd.GeoSeries([Point(X, Y)], crs=aoi.crs)
    point_4326 = point.to_crs(epsg=4326)

    aoi = gpd.read_file(aoi_path)
    aoi_4326 = aoi.to_crs(epsg=4326)

    lat = point_4326.y.iloc[0]
    lon = point_4326.x.iloc[0]
    planner = AOIRoutePlanner(aoi_4326,(lon,lat),'drive',outfolder)
    planner.get_graph_with_buffer()

    end_points_initial = planner.compute_end_points()
    end_points_final = planner.retrieve_by_hierarchy(end_points_initial)

    routes = planner.compute_routes(end_points_final)
    routes = enforceOutputDtypes(routes, INT32_OUTPUT_COLUMNS)
    routes_crs = routes.to_crs(aoi.crs)
    routes_crs = routes_crs.drop(columns='AOI')
    routes_crs['AOI_name'] = aoi['AOI'].iloc[0]
    routes_crs.to_file(os.path.join(outfolder,'evacuation_routes_line_01_intermediate.shp'))
    routes_crs.to_file(os.path.join(outfolder,'evacuation_routes_line_02_edit.shp'))

    point_directions = planner.points_at_thirds_with_direction(routes)
    point_directions = enforceOutputDtypes(point_directions, INT32_OUTPUT_COLUMNS)
    point_directions_crs = point_directions.to_crs(aoi.crs)
    point_directions_crs['AOI_name'] = aoi['AOI'].iloc[0]
    point_directions_crs.to_file(os.path.join(outfolder,'evacuation_routes_point_01_intermediate.shp'))
    point_directions_crs.to_file(os.path.join(outfolder,'evacuation_routes_point_02_edit.shp'))


def enforceOutputDtypes(gdf, int32_cols=None, float32_cols=None, str_cols=None):
    """
    Enforces GIS-safe output dtypes so downstream QA and shapefile writers
    (Fiona/OGR) see the expected field types, instead of pandas' defaults
    (int64 -> "Integer64"/"long int", float64 -> "double").

    Parameters:
    - gdf: The GeoDataFrame whose columns should be retyped.
    - int32_cols: Column names that must become 32-bit integers. Values are
      coerced numerically first (handles strings/floats/NaN-as-object
      columns like those from safe_convert_to_int), then any remaining NaNs
      are filled with 0 before casting, since int32 cannot hold NaN. Rows
      with invalid values should already have been filtered out by this
      point by the relevant handler.
    - float32_cols: Column names that must become 32-bit floats.
    - str_cols: Column names that must become text (e.g. flag fields).

    Returns:
    - gdf: The GeoDataFrame with the requested columns retyped. Columns not
      present in gdf are silently skipped, so this is safe to call with a
      shared list across subproduct types that don't all have the same
      columns.
    """
    if int32_cols:
        for col in int32_cols:
            if col in gdf.columns:
                before_na = gdf[col].isna().sum() if gdf[col].dtype == object else 0
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
                coerced_na = gdf[col].isna().sum() - before_na
                if coerced_na > 0:
                    logger.warning(
                        f"Column '{col}': {coerced_na} value(s) could not be "
                        f"converted to a number and were set to 0 before casting to int32."
                    )
                gdf[col] = gdf[col].fillna(0).astype('int32')

    if float32_cols:
        for col in float32_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce').astype('float32')

    if str_cols:
        for col in str_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].astype(str)

    return gdf


if __name__ == "__main__":
    run_evacuation()
