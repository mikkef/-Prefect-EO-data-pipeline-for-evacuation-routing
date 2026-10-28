#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:19:24 2024

@author: chlorokostas
"""
import os
import codecs
import requests
import osm2geojson
from tqdm import tqdm
import json
import datetime
import geopandas as gpd
import xml.etree.ElementTree as ET
from . import IRIDEClasses
from . import IRIDECodes


class OSM_retriever:

    useragent = 'IRIDE Human Settlements'
    req_header = {
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Google Chrome 80"',
        'Accept': '*/*',
        'Sec-Fetch-Dest': 'empty',
        'User-Agent': useragent,
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://overpass-turbo.eu',
        'Sec-Fetch-Site': 'cross-site',
        'Sec-Fetch-Mode': 'cors',
        'Referer': 'https://overpass-turbo.eu/',
        'Accept-Language': '',
        'dnt': '1',
    }
    bbox = None
    query = None
    response = None
    req_data = None
    req_content = None
    saved_osm_file = None
    saved_geojson = None
    out_format = None
    minlon = None
    minlat = None
    maxlon = None
    maxlat = None
    status = None
    dt_now = datetime.datetime.now()
    dt_suffix = dt_now.strftime('%d%m%y-%H%M%S%MS')
    

    def __init__(self, minlat: float, minlon: float, maxlat: float, maxlon: float, out_format='xml', timeout=600):
        """
        Initiates an OSM_retriever instance to retrieve osm data layers for the defined bounding box.
        Input:
            minlat (float): south latitude of the bounding box
            minlon (float): west longitude of the bounding box
            maxlat (float): north latitude of the bounding box
            maxlon (float): east longitude of the bounding box
            out_format ("xml", "json"): output format, xml or json. Default is XML
            timeout (int): seconds until request times out. Default: 600s
        """
        self.minlat = minlat      
        self.minlon = minlon
        self.maxlat = maxlat
        self.maxlon = maxlon
        
        # bbox = f'{39.12},{26.52},{39.14},{26.55}' # Sample bbox for Kara Tepe
        # bbox = f'{34.621},{10.592},{35.285},{11.236}' # Sample bbox for Tunisia
        # bbox = f'{40.815},{14.091},{40.848},{14.150}' # Sample bbox for Italy
        # Rome test 41.65,12.20,41.89,12.48
        self.bbox = f'{minlat},{minlon},{maxlat},{maxlon}'
        
        # Spatial query
        self.query = f'[out:{out_format}][timeout:{timeout}]; ( node({self.bbox}); way(bn); node(w); ); out meta;'
        #self.query = f'[out:{out_format}][timeout:{timeout}]; ( node({self.bbox}); way(bn); node(w); ); (._;(rel(bn)->.a; rel(bw)->.a;);); out meta;'
        #self.query = f'[out:{out_format}][timeout:600];(nwr({bbox});); out;' # bbox : (south lat, west lon, north lat, east lon)
    
        # Request body
        self.req_data = {
          'data': self.query
        }
        
        self.out_format = out_format
        print('OSM retriever initialized:')
        print(f'Active bounding box: {self.bbox}')
        print(f'Query: {self.query}')
        

    def get_osm_data(self, outfolder=None, verbose=False):
        """
            Recovers osm data layers for the defined bounding box.
            Input:
                outfolder (path, default: None): folder location for the output. Default is the current working directory
            Return:
                1. .osm data saved to a file in XML or JSON format 
                2. overpass request content. Containing nodes, ways and relations
        """    
        
        # request and error catching
        print('Executing Overpass API request.Endpoint: https://overpass-api.de/api/interpreter', end = '\n', flush=True)
        print(f'Active bounding box: {self.bbox}', end = '\n', flush=True)
        print(f'Query: {self.query}', end = '\n', flush=True)

        try:
            # post request to overpass API
            self.response = requests.post('https://overpass-api.de/api/interpreter', headers=self.req_header, data=self.req_data, stream=True)
            self.response.raise_for_status()
            self.status = self.response.status_code
            print(f'Success! Response status: {self.response.status_code}\nTo view response content type <retriever_name>.response.text', end = '\n', flush=True)
        except requests.exceptions.HTTPError as e:
            print(f'HTTPError. Response status: {self.response.status_code}', end = '\n')
            print(e.args[0], flush=True)
            exit()
        except requests.exceptions.ConnectionError:
            print(f'ConnectionError! API request failed. Response status: {self.response.status_code}', end = '\n', flush=True)
            exit()
        except requests.exceptions.ReadTimeout:
            print('Time out. The request timed out while trying to connect to the remote server.', end = '\n', flush=True)
            exit()
        print('Saving data to disk...', end = '\n', flush=True)
        self.req_content = self.response.text
        # Sizes in bytes.
        # total_size = int(self.response.headers.get("content-length", 0))
        total_size = int(len(self.response.content))
        block_size = 1024

        if outfolder is None:
            outfolder=os.getcwd()
        else:
            if not os.path.exists(outfolder):
                raise Exception("Output path does not exist", end = '\n', flush=True)
        
        # export raw overpass data to .json
        if self.out_format == 'json':
            
            outfile = os.path.join(outfolder,f'output_{self.dt_suffix}.json')
            if verbose:
                # saving request content with progress bar tracking
                with open(outfile, mode="wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        for data in self.response.iter_content(block_size):
                            progress_bar.update(len(data))
                            json.dump(data,f)
                print(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}', end = '\n', flush=True)
                
            else:
                # saving request content
                with open(outfile, mode="w") as f:
                    json.dump(self.response.text,f)
                    print(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}', end = '\n', flush=True)
                    
        # export raw overpass data to .osm
        if self.out_format == 'xml':

            outfile = os.path.join(outfolder,f'output_{self.dt_suffix}.osm')
            if verbose:
                # saving request content with progress bar tracking
                with open(outfile, "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        for data in self.response.iter_content(block_size):
                            progress_bar.update(len(data))
                            f.write(data)
            else:
                # saving request content
                with codecs.open(outfile, "wb", 'utf-8') as f:
                    f.write(self.response.text)
            
            tree = ET.parse(outfile)
            root = tree.getroot()
            
            # add a "bounds" element into the xml. Makes it readable by QGIS
            bounds = ET.Element("bounds")
            bounds.attrib["minlat"] = str(self.minlat)
            bounds.attrib["minlon"] = str(self.minlon)
            bounds.attrib["maxlat"] = str(self.maxlat)
            bounds.attrib["maxlon"] = str(self.maxlon)
            root.insert(0,bounds)
            
            tree.write(outfile)
            
            print(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}', end = '\n', flush=True)
            
        self.saved_osm_file = outfile

    def to_geojson(self, outfolder=None, out=False):
        """
            Converts the overpass request content to GeoJSON format.
            Input:
                outfolder (path, default: None): folder location for the output. Default is the current working directory
                out (boolean, default: True): controls weather the function returns the GeoJSON content. If false nothing is returned.
            Return:
                1. osm data saved to a file in GeoJSON format 
                2. GeoJSON string
        """   
        try:
            if self.out_format == 'xml':
                print('Converting OSM XML to GeoJSON...', end = ' ', flush=True)
                
                geojson_out = osm2geojson.xml2geojson(self.req_content)
                print("Done.", end = '\n', flush=True)
                
            
            if self.out_format == 'json':
                print('Converting OSM JSON to GeoJSON...', end = ' ', flush=True)
                
                geojson_out = osm2geojson.json2geojson(self.req_content)
                print("Done.", end = '\n', flush=True)
                
                
            # Isolate all unique "properties" keys from geojson dict and its encapsulated dicts
            print('Curating feature properties tags...', end = ' ', flush=True)
            

            for feature in geojson_out['features']:
                featureprop = feature['properties']
                
                tags = []
                values = []
                # iterate over keys and values of input geojson dict
                for k,v in featureprop.items():
                    
                    # check if value is a nested dict
                    if type(v)==dict:
                        # iterate over keys and values of nested dict
                        for k2,v2 in v.items():
                            # check if key already exists
                            if k2 in tags:
                                continue
                            else:
                                #append new keys/values to their lists
                                tags.append(k2)
                                values.append(v2)
                    else:
                        if k in tags or k=='nodes':
                            continue
                        else:
                            #append new keys/values to their lists
                            tags.append('@'+k)
                            values.append(v)
                            
                new_feat = dict(zip(tags,values))
                new_feat['@id'] = new_feat['@type'] + '/' + str(new_feat['@id'])
                del new_feat['@type']
                
                feature['properties'] = new_feat
            print('Done.', flush=True)
            
            
            if outfolder is None:
                outfolder=os.getcwd()
            else:
                if not os.path.exists(outfolder):
                    raise Exception("Output path does not exist")
     
            outfile = os.path.join(outfolder,f'output_{self.dt_suffix}.geojson')
            with open(outfile, mode="w") as f:
                json.dump(geojson_out,f)
                print(f'The OSM layers corresponding to the input bounding box were saved to: {outfile}', end='\n', flush=True)
                

        except TypeError:
            print('No input data available from OSM API request. Execute a new request using .get_osm_data method.', end='\n', flush=True)
            exit()

        self.saved_geojson = outfile
        
        if out:
            return geojson_out
    
    def extract_roads(self, outfolder=None, out=False):
        """
            Extracts road elements from the OSM response content. This function uses the existing response content 
            from .get_osm_data() method as input. If the dataset is in its raw xml/json form, and has not been converted to GeoJSON,
            it first converts it into GeoJSON using the .to_geojson() method. Afterwards it generates a geopandas dataframe
            from the GeoJSON and queries for any table rows corresponding to highways, bridges and tunnels. Attributes corresponding
            to the IRIDE nomeclature are added and their values for each featyre are updated accordingly.
            The result is exported in a separate GeoJSON file.
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
        print('Extracting road network data...', end = ' ', flush=True)  

        if self.saved_geojson is not None:      # check if GeoJSON conversion has been done  
            gdf = gpd.read_file(self.saved_geojson)     # Read OSM data from GeoJSON file into a geopandas dataframe
        
        else:
            if self.req_content == None:        # check if request data exist
                print('No input data available from OSM API request. Executing a new request using .get_osm_data method.', end='\n', flush=True)
                self.get_osm_data()
            
            geojson_out = self.to_geojson(out=True) # Convert request content to GeoJSON
            
            gdf = gpd.GeoDataFrame.from_features(geojson_out['features']) # Read GeoJSON to a geopandas dataframe
            del geojson_out     # Free up memory
        
        # isolate Road Network features (roads, tunnels and bridges)       
        RoadN = gdf.loc[((gdf['highway'].notnull()) & (gdf.geom_type=='LineString'))].copy(deep=True)

        del gdf        # Free up memory, gdf won't be used again
        
        if len(RoadN.index)==0:
            print('No road network data found for the input Area of Interest.', end='\n', flush=True)
            
        RoadN.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
        
        # Add or rename columns to match the IRIDE nomeclature
        if 'bridge' in RoadN.columns:
            RoadN.rename(columns={"bridge":"RNbridge"}, inplace=True)
        if 'tunnel' in RoadN.columns:
            RoadN.rename(columns={"tunnel": "RNtunnel"}, inplace=True)
        if 'lanes' in RoadN.columns:
            RoadN.rename(columns={"lanes": "RNLanes"}, inplace=True)
        if 'oneway' in RoadN.columns:
            RoadN.rename(columns={"oneway": "RNOneWay"}, inplace=True)
        RoadN.insert(2,"RNCode",9991)
        RoadN.insert(3, "RNClass", 'None')
        
        # Update road column values according to the IRIDE nomeclature
        for k in RoadN['highway'].unique():
            if k in IRIDECodes.keys():
                RoadN.loc[RoadN.highway == str(k), 'RNCode'] = IRIDECodes[str(k)]
            else:
                continue
            
        for k in RoadN['RNCode'].unique():
                RoadN.loc[RoadN.RNCode == k, 'RNClass'] = IRIDEClasses[str(k)]
            
        RoadN['RNLanes'] = RoadN['RNLanes'].fillna(0)
        RoadN['RNOneWay'] = RoadN['RNOneWay'].fillna(0)
        RoadN['RNLanes'] = RoadN['RNLanes'].fillna(0)
        if 'RNbridge' in RoadN.columns:
            RoadN['RNbridge'] = RoadN['RNbridge'].fillna(0)
        if 'RNtunnel' in RoadN.columns:
            RoadN['RNtunnel'] = RoadN['RNtunnel'].fillna(0)
        print('Done!', end='\n', flush=True)
        

        # GeoJSON exort
        if outfolder is None:
            outfolder=os.getcwd()
        else:
            if not os.path.exists(outfolder):
                raise Exception("Output path does not exist")
 
        outfile = os.path.join(outfolder,f'RoadN_{self.dt_suffix}.geojson')
        
        RoadN.to_file(outfile, driver='GeoJSON')
        
        print(f'Road Network features extracted and saved to: {outfile}', end='\n', flush=True)
        
        
        if out == 'gdf':
            return RoadN
        elif out == 'geojson':
            return RoadN.to_json()


    def extract_rail(self, outfolder=None, out=False):
        """
            Extracts rail elements from the OSM response content. This function uses the existing response content 
            from .get_osm_data() method as input. If the dataset is in its raw xml/json form, and has not been converted to GeoJSON,
            it first converts it into GeoJSON using the .to_geojson() method. Afterwards it generates a geopandas dataframe
            from the GeoJSON and queries for any table rows corresponding to highways, bridges and tunnels. Attributes corresponding
            to the IRIDE nomeclature are added and their values for each featyre are updated accordingly.
            The result is exported in a separate GeoJSON file.
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
        print('Extracting rail network data...', end = ' ', flush=True)
        
        if self.saved_geojson is not None:      # check if GeoJSON conversion has been done  
            gdf = gpd.read_file(self.saved_geojson)     # Read OSM data from GeoJSON file into a geopandas dataframe
        
        else:
            if self.req_content == None:        # check if request data exist
                print('No input data available from OSM API request. Executing a new request using .get_osm_data method.', end='\n', flush=True)
                self.get_osm_data()
            
            geojson_out = self.to_geojson(out=True) # Convert request content to GeoJSON
            
            gdf = gpd.GeoDataFrame.from_features(geojson_out['features']) # Read GeoJSON to a geopandas dataframe
            del geojson_out     # Free up memory
        
        print('Extracting rail network data...', end = ' ', flush=True)
        
        # isolate Railways features
        RailN = gdf.loc[((gdf['railway'].notnull()) & (gdf.geom_type=='LineString'))].copy(deep=True)

        del gdf        # Free up memory, gdf won't be used again
        
        if len(RailN.index)==0:
            print('No railroad network data found for the input Area of Interest.', end='\n', flush=True)
            

        RailN.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
        
        # Add or rename columns to match the IRIDE nomeclature
        if 'bridge' in RailN.columns:
            RailN.rename(columns={"bridge":"RNbridge"}, inplace=True)
        if 'tunnel' in RailN.columns:
            RailN.rename(columns={"tunnel": "RNtunnel"}, inplace=True)
        RailN.insert(2,"RNCode",9991)
        RailN.insert(3, "RNClass", 'None')
        
        # Update road column values according to the IRIDE nomeclature
        for k in RailN['railway'].unique():
            if k in IRIDECodes.keys():
                RailN.loc[RailN.railway == str(k), 'RNCode'] = IRIDECodes[str(k)]
            else:
                continue
            
        for k in RailN['RNCode'].unique():
            RailN.loc[RailN.RNCode == k, 'RNClass'] = IRIDEClasses[str(k)]

        if 'RNbridge' in RailN.columns:
            RailN['RNbridge'] = RailN['RNbridge'].fillna(0)
        if 'RNtunnel' in RailN.columns:
            RailN['RNtunnel'] = RailN['RNtunnel'].fillna(0)
        print('Done!', flush=True)
        

        # GeoJSON exort
        if outfolder is None:
            outfolder=os.getcwd()
        else:
            if not os.path.exists(outfolder):
                raise Exception("Output path does not exist")
 
        outfile = os.path.join(outfolder,f'RailN_{self.dt_suffix}.geojson')
        
        RailN.to_file(outfile, driver='GeoJSON')
        
        print(f'Rail Network features extracted and saved to: {outfile}', end='\n', flush=True)
        

        if out == 'gdf':
            return RailN
        elif out == 'geojson':
            return RailN.to_json()

    def extract_hydro(self, outfolder=None, out=False):
            """
                Extracts Hydrography Network elements from the OSM response content. This function uses the existing response content 
                from .get_osm_data() method as input. If the dataset is in its raw xml/json form, and has not been converted to GeoJSON,
                it first converts it into GeoJSON using the .to_geojson() method. Afterwards it generates a geopandas dataframe
                from the GeoJSON and queries for any table rows corresponding to highways, bridges and tunnels. Attributes corresponding
                to the IRIDE nomeclature are added and their values for each featyre are updated accordingly.
                The result is exported in a separate GeoJSON file.
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
            if self.saved_geojson is not None:      # check if GeoJSON conversion has been done  
                gdf = gpd.read_file(self.saved_geojson)     # Read OSM data from GeoJSON file into a geopandas dataframe
            
            else:
                if self.req_content == None:        # check if request data exist
                    print('No input data available from OSM API request. Executing a new request using .get_osm_data method.', end='\n', flush=True)
                    self.get_osm_data()
                
                geojson_out = self.to_geojson(out=True) # Convert request content to GeoJSON
                
                gdf = gpd.GeoDataFrame.from_features(geojson_out['features']) # Read GeoJSON to a geopandas dataframe
                del geojson_out     # Free up memory
            
            print('Extracting hydrography network data...', end = ' ', flush=True)
            
            # isolate Railways features
            HydroN = gdf.loc[((gdf['waterway'].notnull()) & (gdf.geom_type=='LineString'))].copy(deep=True)

            del gdf        # Free up memory, gdf won't be used again
            
            if len(HydroN.index)==0:
                print('No hydrographic network data found for the input Area of Interest.', end='\n', flush=True) 
                

            HydroN.dropna(axis=1,how='all',inplace=True)    # Prune columns with no data
            
            # Add or rename columns to match the IRIDE nomeclature
            if 'name' in HydroN.columns:
                HydroN.rename(columns={"name":"nameEN"}, inplace=True)
            HydroN.insert(2,"HNCode",9991)
            HydroN.insert(3, "HNClass", 'None')
            
            # Update road column values according to the IRIDE nomeclature
            for k in HydroN['waterway'].unique():
                if k in IRIDECodes.keys():
                    HydroN.loc[HydroN.waterway == str(k), 'HNCode'] = IRIDECodes[str(k)]
                else:
                    continue
                
            for k in HydroN['HNCode'].unique():
                HydroN.loc[HydroN.HNCode == k, 'HNClass'] = IRIDEClasses[str(k)]

            print('Done!', end='\n', flush=True)

            # GeoJSON exort
            if outfolder is None:
                outfolder=os.getcwd()
            else:
                if not os.path.exists(outfolder):
                    raise Exception("Output path does not exist")
    
            outfile = os.path.join(outfolder,f'HN_{self.dt_suffix}.geojson')
            
            HydroN.to_file(outfile, driver='GeoJSON')
            
            print(f'Hydrography Network features extracted and saved to: {outfile}', end='\n', flush=True)
            

            if out == 'gdf':
                return HydroN
            elif out == 'geojson':
                return HydroN.to_json()
