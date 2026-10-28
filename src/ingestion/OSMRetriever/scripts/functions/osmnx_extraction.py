#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:19:24 2024

@author: chlorokostas
"""
import os
import osmnx
import datetime
import geopandas as gpd
import pandas as pd
from loguru import logger
from osmnx import features_from_polygon
from src.ingestion.OSMRetriever.scripts.functions.exceptions import BBOXError, DriverError, CRSError
from pathlib import Path
# OSM layer nomenclature to IRIDE nomeclature translation

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class OSM_retriever:

    saved_geojson = None
    bbox=None
    minlon = None
    minlat = None
    maxlon = None
    maxlat = None
    aoiname = None
    dt_now = datetime.datetime.now()
    dt_suffix = dt_now.strftime('%d%m%y-%H%M%S%MS')
    data = None
    tags = {
            'highway': True,
            'building': True,
            'waterway': True,
            'railway': True,
            'cables':True,
            'place': True,
            'amenity': True,
            'aeroway': True,
            'leisure': True,
             'natural': True,
            'reservoir':True,
            'lake': True,
            }   

    def __init__(self, bbox:str=None):
        """
        Initiates an OSM_retriever instance to retrieve osm data layers for the defined bounding box.
        Input:
            bbox (string): Bounding box parameters as a string '{minlat},{minlon},{maxlat},{maxlon}' where
                           minlat, minlon, maxlat, maxlon are float numbers.

        """
        if bbox==None:

            logger.info('OSM retriever initialized.')
            logger.info(f'No bounding box defined! Either define one through .bbox method or read an existing osm .geojson with .read_from_file method.')

        else:
            try:
                if not isinstance(bbox, str):
                    raise TypeError("Input Bounding Box is not a string of 'minlat,minlon,maxlat,maxlon' format.")
            except TypeError as e:
                logger.error(e)
                return
            
            bounds = bbox.split(',')
            try:
                if len(bounds)<4:
                    raise BBOXError(bbox)
                if False in (is_number(t) for t in bounds):
                    raise BBOXError(bbox)
                if not (-90<=float(bounds[0])<=90) & (-90<=float(bounds[2])<=90) & (-180<=float(bounds[1])<=180) & (-180<=float(bounds[3])<=180):
                    raise BBOXError(bbox)
                else:
                    self.bbox = bbox
                    self.minlat = bounds[0]
                    self.minlon = bounds[1] 
                    self.maxlat = bounds[2]
                    self.maxlon = bounds[3]
                    self.aoiname = f'{self.minlat}_{self.minlon}_{self.maxlat}_{self.maxlon}'

                    logger.info('OSM retriever initialized:')
                    logger.info(f'Active bounding box: {self.bbox}')
            
            except BBOXError as e:
                logger.error(e)
    
    def define_bbox(self, bbox:str=None, aoi:str=None):
        """
        Defines bounding box for the OSM_retriever.
        Input:
            bbox (string): Bounding box parameters as a string '{minlat},{minlon},{maxlat},{maxlon}' where
                           minlat, minlon, maxlat, maxlon are float numbers.
            aoi (path): Path to bucket_data file (either .shp or .geojson)
        """

        if bbox:
            try:
                if not isinstance(bbox, str):
                    raise TypeError("Input Bounding Box is not a string of 'minlat,minlon,maxlat,maxlon' format.")
            except TypeError as e:
                logger.error(e)
                return
            
            bounds = bbox.split(',')
            try:
                if len(bounds)<4:
                    raise BBOXError(bbox)
                if False in (is_number(t) for t in bounds):
                    raise BBOXError(bbox)
                if not (-90<=float(bounds[0])<=90) & (-90<=float(bounds[2])<=90) & (-180<=float(bounds[1])<=180) & (-180<=float(bounds[3])<=180):
                    raise BBOXError(bbox)
                else:
                    self.bbox = bbox
                    self.minlat = bounds[0]
                    self.minlon = bounds[1] 
                    self.maxlat = bounds[2]
                    self.maxlon = bounds[3]
                    self.aoiname = f'{self.minlat}_{self.minlon}_{self.maxlat}_{self.maxlon}'
                    logger.info(f'Active bounding box: {self.bbox}')

            except BBOXError as e:
                logger.error(e)

        elif aoi:
            try:
                if not isinstance(aoi, str):
                    raise TypeError("Input bucket_data path is not a string.")
            except TypeError as e:
                logger.error(e)
                return
            
            if not os.path.exists(aoi):
                logger.error(f'No bucket_data file found at path: "{aoi}"!')
                return
            else:
                try:
                    logger.info(f'Reading bucket_data from file: "{aoi}"...')
                    aoi_gdf = gpd.read_file(aoi)
                    if not aoi_gdf.crs=='EPSG:4326':
                        logger.info(f'Reprojecting bucket_data bounds to WGS 84 ...')
                        aoi_gdf.to_crs(crs=4326, inplace=True)
                    bounds = aoi_gdf.total_bounds
                    self.minlat = bounds[1]      
                    self.minlon = bounds[0] 
                    self.maxlat = bounds[3]
                    self.maxlon = bounds[2]
                    self.bbox = f'{self.minlat},{self.minlon},{self.maxlat},{self.maxlon}'
                    logger.info(f'Active bounding box: {self.bbox}')
                    basename = os.path.basename(aoi)
                    self.aoiname = basename.split('.')[0]
                except DriverError as e:
                    logger.error(e)
                    return

        else:
            logger.info(f'Bounding Box was not defined in initialization.')
            return
    

    def read_from_file(self, input:str):

        try:
            if not isinstance(input, str):
                raise TypeError("Input path is not a string.")
        except TypeError as e:
            logger.error(e)
            return
        
        if not os.path.exists(input):
            logger.error(f'No .geojson/.shp file found at {input}!')
            return
        else:
            logger.info('Loading vector data...')
            try:
                self.data = gpd.read_file(input)
                self.data.drop(self.data.loc[:,(self.data.dtypes=='datetime64[ns]')],axis=1, inplace=True)
                bounds = self.data.total_bounds
                self.minlat = bounds[1]      
                self.minlon = bounds[0] 
                self.maxlat = bounds[3]
                self.maxlon = bounds[2]
                self.bbox = f'{self.minlat},{self.minlon},{self.maxlat},{self.maxlon}'
                logger.info(f'Data loaded and is now accessible through the <retriever_name>.data method.')
                logger.info(f'Active bounding box: {self.bbox}')
            except DriverError as e:
                    logger.error(e)
                    return

    def get_osm_data(self, export:bool=False, outfolder:str=None, crs_code:int=None, driver:str = 'geojson'):
        """
            Recovers osm data layers for the defined bounding box.
            Input:
                export (boolean): If True, it exports the retrieved OSM data into a .geojson file at the folder
                                  location defined in "outfolder" variable.
                outfolder (path, default: None): folder location for the output. Default is the current working directory
                crs_code (int): ESPG code input for data reprojection.
            Return:
                1. geojson file containing osm data
                2. geopandas frame containing osm data
        """    
        
        if not self.bbox:
            logger.error('A bbox has not been defined during initialization or through the .define_bbox method.')
            return

        logger.info(f'Retrieving OSM data for bbox = {self.bbox} ...')

        self.data = osmnx.features_from_bbox(bbox=(self.minlon,self.minlat,self.maxlon,self.maxlat),tags=self.tags)
        logger.info('Data retrieval completed.')

        # Drop Datetime Columns
        self.data.drop(self.data.loc[:,(self.data.dtypes=='datetime64[ns]')],axis=1, inplace=True)

        if crs_code!=None and crs_code!=4326:
            try:
                if not isinstance(crs_code, int):
                    raise TypeError("Input CRS code is not an integer.")
            except TypeError as e:
                logger.error(e)
                return

            try:
                logger.info(f'Reprojecting dataset from EPSG:4326 to EPSG:{crs_code}')
                target_crs = f'epsg:{crs_code}'
                reprojected= self.data.to_crs(target_crs)
                self.data = reprojected.copy(deep=True)
                logger.info('Done')
            except CRSError as e:
                logger.error(e)
                return
        
        if export:
            if outfolder is None:
                outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
                if not os.path.exists(outfolder):
                    os.mkdir(outfolder)
            else:
                try:
                    if not isinstance(outfolder, str):
                        raise TypeError("Output path is not a string.")
                except TypeError as e:
                    logger.error(e)
                    return

                if not os.path.exists(outfolder):
                    logger.error("Output path does not exist.")

            outfolder=os.path.join(outfolder,self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            outfile = os.path.join(outfolder,f'output_{self.dt_suffix}.{driver}')
            self.data.to_file(outfile)
            
            logger.info(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}')
            self.saved_geojson = outfile
    
    def save_to_file(self, outfolder=None, crs_code: int = None, driver:str = 'geojson'):
        
        if crs_code!=None and crs_code!=4326 :
            try:
                logger.info(f'Reprojecting dataset from EPSG:4326 to EPSG:{crs_code}')
                target_crs = f'epsg:{crs_code}'
                reprojected= self.data.to_crs(target_crs)
                self.data = reprojected.copy(deep=True)
                logger.info('Done')
            except CRSError as e:
                logger.error(e)
        
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            logger.error("Output path does not exist")
            return

        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'output_{self.dt_suffix}.{driver}')
        self.data.to_file(outfile)

        logger.info(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}')
        self.saved_geojson = outfile

    def extract_roads(self, outfolder=None, out=False, driver: str = 'shp', aoi=None, logger=None):
        """
            Extracts road elements from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                              Default is 'None', function returns nothing.
            Return:
                1. Osm road network data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """
        if self.data is None:  # check if data are available
            logger.error(
                'No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif 'highway' not in self.data.columns:
            logger.info('No road network data found for the input Area of Interest.')
            return
        else:
            logger.info('Extracting road network data...')
            # isolate Road Network features (roads, tunnels and bridges)       
            RoadN = self.data.loc[((self.data['highway'].notnull()) & (self.data.geom_type == 'LineString'))].copy(
                deep=True)

        RoadN.dropna(axis=1, how='all', inplace=True)  # Prune columns with no data

        logger.info('Done!')

        aoi_gdf = gpd.read_file(aoi)
        RoadN = RoadN.to_crs(aoi_gdf.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder = os.path.join(os.getcwd(), 'outputs', self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            logger.error("Output path does not exist")
            return

        # outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)

        # Extract coordinates and CRS from the AOI filename
        parts = Path(aoi).stem.split("_")
        lat, lon, crs = parts[1], parts[2], parts[3]

        # Create output filename
        outfile = Path(outfolder) / f"roads_{lat}_{lon}_{crs}.shp"
        RoadN.to_file(outfile)

        logger.info(f'Road Network features extracted and saved to: {outfile}')

        return outfile
        # if out == 'gdf':
        #     return outfile
        # elif out == 'geojson':
        #     return RoadN.to_json()

    def extract_rail(self, input=None, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts rail elements from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                              Default is 'None', function returns nothing.
            Return:
                1. Osm rail network data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """
        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif 'railway' not in self.data.columns:
            logger.info('No railroad network data found for the input Area of Interest.')
            return
        else:
            logger.info('Extracting rail network data...')
            # isolate Railways features
            RailN = self.data.loc[((self.data['railway'].notnull()) & (self.data.geom_type=='LineString'))].copy(deep=True)

        RailN.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data

        aoi = gpd.read_file(aoi)
        RailN = RailN.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            logger.error("Output path does not exist")
            return

        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'rail_network_01_retrieved.{driver}')
        RailN.to_file(outfile)

        logger.info(f'Rail Network features extracted and saved to: {outfile}')

        if out == 'gdf':
            return RailN
        elif out == 'geojson':
            return RailN.to_json()

    def extract_hydro(self, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts Hydrography Network elements from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                            Default is 'None', function returns nothing.
            Return:
                1. Osm hydrography network data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """   
        
        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif 'waterway' not in self.data.columns and 'natural' not in self.data.columns:
            logger.info('No hydrographic network data found for the input Area of Interest.') 
            return
        else:
            logger.info('Extracting hydrography network data...')
            # isolate waterway features
            if 'waterway' in self.data.columns and 'natural' in self.data.columns:
                HN = self.data.loc[(((self.data['waterway'].notnull()) | (self.data['natural']=='coastline')) & (self.data.geom_type=='LineString'))].copy(deep=True)
                Lakes = self.data.loc[(((self.data['natural']=='water') & ((self.data['water'] =='reservoir') | (self.data['water'] =='lake'))) & (self.data.geom_type=='Polygon'))].copy(deep=True)
            if 'waterway' in self.data.columns and 'natural' not in self.data.columns:
                HN = self.data.loc[((self.data['waterway'].notnull()) & (self.data.geom_type=='LineString'))].copy(deep=True)
            if 'waterway' not in self.data.columns and 'natural' in self.data.columns:
                HN = self.data.loc[((self.data['natural']=='coastline') & (self.data.geom_type=='LineString'))].copy(deep=True)
                Lakes = self.data.loc[(((self.data['natural']=='water') & ((self.data['water'] =='reservoir') | (self.data['water'] =='lake'))) & (self.data.geom_type=='Polygon'))].copy(deep=True)

        # Curate data for Lakes and reservoirs
        if Lakes.shape[0]==0:
            logger.info('No data found for lakes or reservoirs.')


        # Merge HN and Lakes
        Merged = gpd.GeoDataFrame(pd.concat([HN,Lakes],axis=0))
        Merged.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data

        logger.info('Done!')

        aoi = gpd.read_file(aoi)
        Merged = Merged.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            logger.error("Output path does not exist")
            return
        
        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'hydrography_01_retrieved.{driver}')
        Merged.to_file(outfile)
        
        logger.info(f'Hydrography Network features extracted and saved to: {outfile}')

        if out == 'gdf':
            return Merged
        elif out == 'geojson':
            return Merged.to_json()
        
    def extract_buildings(self, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts Building elements from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                            Default is 'None', function returns nothing.
            Return:
                1. Osm building data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """   
        
        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif 'building' not in self.data.columns:
            logger.info('No building data found for the input Area of Interest.') 
            return
        else:
            logger.info('Extracting building data...')
            # isolate building features
            HS = self.data.loc[((self.data['building'].notnull()) & ((self.data.geom_type=='Polygon') | (self.data.geom_type=='Multipolygon')))].copy(deep=True)
            
        HS.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
        logger.info('Done!')

        aoi = gpd.read_file(aoi)
        HS = HS.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            logger.error("Output path does not exist")
        
        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'human_settlements_01_retrieved_osm.{driver}')
        HS.to_file(outfile)
        
        logger.info(f'Building features extracted and saved to: {outfile}')
        

        if out == 'gdf':
            return HS
        elif out == 'geojson':
            return HS.to_json()
        
    def extract_places(self, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts Populated and Natural places from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                            Default is 'None', function returns nothing.
            Return:
                1. Osm populated places data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """   

        if self.data is None or self.data.empty:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif not self.data.columns.isin(['place', 'natural']).any():
            logger.info('No Places data found for the input Area of Interest.')
            return
        else:
            logger.info('Extracting Places data...')

            if 'place' not in self.data.columns:
                self.data['place'] = ''

            PP = self.data.loc[(self.data['place'].notnull()) & (self.data.geom_type=='Point')].copy(deep=True)
            
        PP.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
        logger.info('Done!')

        aoi = gpd.read_file(aoi)
        PP = PP.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            raise Exception("Output path does not exist")
        
        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'places_01_retrieved.{driver}')
        PP.to_file(outfile)
        
        logger.info(f'Places extracted and saved to: {outfile}')

        if out == 'gdf':
            return PP
        elif out == 'geojson':
            return PP.to_json()
        
    def extract_powergrid(self, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts powergrid from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                            Default is 'None', function returns nothing.
            Return:
                1. Osm powergrid data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """   

        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif 'cables' not in self.data.columns:
            logger.info('No Powergrid data found for the input Area of Interest.') 
            return
        else:
            logger.info('Extracting Powergrid data...')
            # isolate powergrid features
            PG = self.data.loc[(self.data['cables'].notnull()) & (self.data.geom_type=='LineString')].copy(deep=True)

        PG.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data

        logger.info('Done!')
        aoi = gpd.read_file(aoi)
        PG = PG.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            raise Exception("Output path does not exist")
        
        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'PG.{driver}')
        PG.to_file(outfile)
        
        logger.info(f'Powergrid extracted and saved to: {outfile}')
        

        if out == 'gdf':
            return PG
        elif out == 'geojson':
            return PG.to_json()
        
    def extract_poi(self, outfolder=None, out=False, driver:str = 'shp', aoi=None):
        """
            Extracts Points of Interest from the OSM response content.
            Input:
                outfolder (path): folder location for the output. Default is the current working directory
                out ('gdf','geojson','None'): If selected the function returns the Road Network features as:
                                                1. 'gdf' geopandas dataframe
                                                2. 'geojson' GeoJSON dict
                                            Default is 'None', function returns nothing.
            Return:
                1. Osm Points of Interest data saved to a file in GeoJSON format, encoded with the IRIDE project nomeclature
                2. GeoJSON string or Geopandas Dataframe
        """   

        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return
        elif not self.data.columns.isin(['amenity', 'leisure','railway','aeroway']).any(): # if no relevant columns exist, quit
            logger.info('No POI data found for the input Area of Interest.')
            return
        else:
            logger.info('Extracting POI data...')
            for i in ['amenity', 'leisure','railway','aeroway']: # if only some columns are missing, add them as empty columns
                if i not in self.data.columns:
                    self.data[i] = ''
            # isolate POI features
            POI = self.data.loc[((self.data['amenity'].notnull()) | (self.data['leisure']=='stadium') | (self.data['railway']=='station') | (self.data['aeroway'].isin(['aerodrome','airfield','helipad']))) & (self.data.geom_type=='Point')].copy(deep=True)
        
        if len(POI.index)==0:
            logger.info('No POI data found for the input Area of Interest.') 
            
        POI.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
        logger.info('Done!')

        aoi = gpd.read_file(aoi)
        POI = POI.to_crs(aoi.crs)

        # GeoJSON export
        if outfolder is None:
            outfolder=os.path.join(os.getcwd(),'outputs',self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

        if not os.path.exists(outfolder):
            raise Exception("Output path does not exist")
        
        outfolder=os.path.join(outfolder,self.aoiname)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        outfile = os.path.join(outfolder,f'POIs_01_retrieved.{driver}')
        POI.to_file(outfile)
        
        logger.info(f'POI data extracted and saved to: {outfile}')
        

        if out == 'gdf':
            return POI
        elif out == 'geojson':
            return POI.to_json()


    def extract_pitches(self, outfolder=None, out=False, driver: str = 'shp', aoi=None):
        if self.data is None:      # check if data are available
            logger.error('No input data available! Either retrieve data from OSM with the .get_osm_data method or read data from an existing file with .read_from_file method.')
            return

        else:
            aoi = gpd.read_file(aoi)
            pitches = features_from_polygon(aoi.to_crs(epsg=4326).geometry.iloc[0], tags={'leisure': 'pitch'})

            # Enforce AND: sport tag must also exist
            if 'sport' in pitches.columns:
                pitches = pitches[pitches['sport'].notna()]
            else:
                pitches = pitches.iloc[0:0]  # empty if no sport column

            pitches = pitches[pitches.geom_type.isin(["Polygon", "MultiPolygon"])]
            pitches = pitches.copy()
            pitches = pitches.to_crs(aoi.crs)

            # GeoJSON export
            if outfolder is None:
                outfolder = os.path.join(os.getcwd(), 'outputs', self.aoiname)
                if not os.path.exists(outfolder):
                    os.mkdir(outfolder)

            if not os.path.exists(outfolder):
                raise Exception("Output path does not exist")

            outfolder = os.path.join(outfolder, self.aoiname)
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            outfile = os.path.join(outfolder, f'pitches_01_retrieved.{driver}')
            pitches.to_file(outfile)

            logger.info(f'POI data extracted and saved to: {outfile}')

            if out == 'gdf':
                return pitches
            elif out == 'geojson':
                return pitches.to_json()
