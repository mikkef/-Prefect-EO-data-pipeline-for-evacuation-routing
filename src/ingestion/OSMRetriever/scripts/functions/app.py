#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:19:24 2024

@author: chlorokostas
"""
        # bbox = f'{39.12},{26.52},{39.14},{26.55}' # Sample bbox for Kara Tepe
        # bbox = f'{34.621},{10.592},{35.285},{11.236}' # Sample bbox for Tunisia
        # bbox = f'{40.815},{14.091},{40.848},{14.150}' # Sample bbox for Italy
        # Rome test 41.65,12.20,41.89,12.48
import os
import click
import osmnx
import subprocess
from src.ingestion.OSMRetriever.scripts.functions.init_logger import logger
from src.ingestion.OSMRetriever.scripts.functions.osmnx_extraction import OSM_retriever


@click.command()
@click.option('-bbox', 'bounding_box', default=None, type=str, help= "Bounding box parameters as a string 'minlat,minlon,maxlat,maxlon' where minlat, minlon, maxlat, maxlon are float numbers.")
@click.option('-aoi', 'input_aoi', default=None, type=click.Path(resolve_path=True), help='Path to aoi geometry (.geojson or .shp). Overrides initialization bbox if both are given.')
@click.option('-o', 'outfolder', type=click.Path(resolve_path=True), default=None, help='Path of output folder for all exports.')
@click.option('-format', type=click.Choice(['geojson','shp']), default='shp', help='Output format of OSM layers. Values: ["geojson","shp"]')
@click.option('-crs', 'crs_out', type=int, default=4326, help='Output EPSG code. ALL layer exports will be reprojected to this EPSG.')
@click.option('-buildings/-no-buildings', 'buildings', default=False, help='Run in debug mode. Catches error traceback.')
@click.option('-roads/-no-roads', 'roads', default=True, help='Extract roads.')
@click.option('-rail/-no-rail', 'rail', default=False, help='Extract rail.')
@click.option('-hydro/-no-hydro', 'hydro', default=False, help='Extract hydro.')
@click.option('-poi/-no-poi', 'poi', default=False, help='Extract poi.')
@click.option('-places/-no-places', 'places', default=False, help='Extract places.')
@click.option('-pitches/-no-pitches', 'pitches', default=False, help='Extract places.')
@click.option('-debug/-no-debug', 'debug', default=False, help='Run in debug mode. Catches error traceback.')

def OSM_to_IRIDE(bounding_box, input_aoi, outfolder, format, crs_out, buildings, roads, rail, hydro, poi, places, pitches, debug):
    if debug:
        with logger.catch():
            retriever = OSM_retriever()
            if not bounding_box==None:
                retriever.define_bbox(bbox=bounding_box)
            if not input_aoi==None:
                retriever.define_bbox(aoi=input_aoi)
            retriever.get_osm_data(crs_code=crs_out, outfolder=outfolder)
            if roads:
                retriever.extract_roads(outfolder=outfolder, driver=format, aoi=input_aoi)
            if rail:
                retriever.extract_rail(outfolder=outfolder, driver=format, aoi=input_aoi)
            if buildings:
                retriever.extract_buildings(outfolder=outfolder, driver=format, aoi=input_aoi)
            if hydro:
                retriever.extract_hydro(outfolder=outfolder, driver=format, aoi=input_aoi)
            if poi:
                retriever.extract_poi(outfolder=outfolder, driver=format, aoi=input_aoi)
            if places:
                retriever.extract_places(outfolder=outfolder, driver=format, aoi=input_aoi)
            if pitches:
                retriever.extract_pitches(outfolder=outfolder, driver=format, aoi=input_aoi)

    else:
        retriever = OSM_retriever()
        if not bounding_box==None:
            retriever.define_bbox(bbox=bounding_box)
        if not input_aoi==None:
                retriever.define_bbox(aoi=input_aoi)
        retriever.get_osm_data(crs_code=crs_out, outfolder=outfolder)
        if roads:
            retriever.extract_roads(outfolder=outfolder, driver=format, aoi=input_aoi)
        if rail:
            retriever.extract_rail(outfolder=outfolder, driver=format, aoi=input_aoi)
        if buildings:
            retriever.extract_buildings(outfolder=outfolder, driver=format, aoi=input_aoi)
        if hydro:
            retriever.extract_hydro(outfolder=outfolder, driver=format, aoi=input_aoi)
        if poi:
            retriever.extract_poi(outfolder=outfolder, driver=format, aoi=input_aoi)
        if places:
            retriever.extract_places(outfolder=outfolder, driver=format, aoi=input_aoi)
        if pitches:
            retriever.extract_pitches(outfolder=outfolder, driver=format, aoi=input_aoi)



def run_OSM_to_IRIDE(
    input_aoi,
    bounding_box=None,
    outfolder=None,
    format='shp',
    crs_out=4326,
    buildings=False,
    roads=True,
    rail=False,
    hydro=False,
    poi=False,
    places=False,
    pitches=False,
    debug=False,
    logger=logger
):
    roads_shp_list = []
    input_aoi = str(input_aoi)
    retriever = OSM_retriever()

    if bounding_box is not None:
        retriever.define_bbox(bbox=bounding_box)

    if input_aoi is not None:
        retriever.define_bbox(aoi=input_aoi)

    retriever.get_osm_data(crs_code=crs_out, outfolder=outfolder)

    if roads:
        roads_shp_path = retriever.extract_roads(outfolder=outfolder, driver=format, aoi=input_aoi, logger=logger)
    if rail:
        retriever.extract_rail(outfolder=outfolder, driver=format, aoi=input_aoi)
    if buildings:
        retriever.extract_buildings(outfolder=outfolder, driver=format, aoi=input_aoi)
    if hydro:
        retriever.extract_hydro(outfolder=outfolder, driver=format, aoi=input_aoi)
    if poi:
        retriever.extract_poi(outfolder=outfolder, driver=format, aoi=input_aoi)
    if places:
        retriever.extract_places(outfolder=outfolder, driver=format, aoi=input_aoi)
    if pitches:
        retriever.extract_pitches(outfolder=outfolder, driver=format, aoi=input_aoi)

    return str(roads_shp_path)

if __name__ == "__main__":

    OSM_to_IRIDE()