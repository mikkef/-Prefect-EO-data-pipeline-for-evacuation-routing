import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import pandas as pd
import math
from loguru import logger
import os
import sys
import time

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("Evacuation routes log.log", rotation="10 MB")

# Fallback Overpass servers, tried after OSMnx's configured default server
FALLBACK_OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
DOWNLOAD_ATTEMPTS_PER_SERVER = 3

class AOIRoutePlanner:
    def __init__(self, aoi_gdf, start_point, network_type,outfolder):
        """
        Initialize the AOIRoutePlanner.

        Parameters
        ----------
        aoi_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing AOI polygon(s) in EPSG:4326.
        start_point : tuple
            Tuple (lon, lat) representing the starting location.
        network_type : str
            Network type for OSMnx (e.g., 'drive', 'walk', 'bike').
        outfolder : str
            Path to output folder for saving intermediate files.
        """
        self.init_graph = None
        self.aoi_gdf = aoi_gdf
        self.start_point = Point(start_point)
        self.network_type = network_type
        self.outfolder = outfolder

    def _download_graph(self, bbox):
        """
        Download the OSMnx graph for a bbox, retrying on failure and falling back
        to other Overpass servers if the default one is busy.

        Parameters
        ----------
        bbox : list
            [minx, miny, maxx, maxy] in EPSG:4326.

        Returns
        -------
        networkx.MultiDiGraph
        """
        default_url = ox.settings.overpass_url
        servers = [default_url] + [u for u in FALLBACK_OVERPASS_SERVERS if u != default_url]
        last_err = None
        try:
            for url in servers:
                ox.settings.overpass_url = url
                for attempt in range(1, DOWNLOAD_ATTEMPTS_PER_SERVER + 1):
                    try:
                        logger.info(f"Downloading graph from {url} (attempt {attempt})")
                        return ox.graph_from_bbox(bbox, network_type=self.network_type)
                    except Exception as e:
                        last_err = e
                        logger.warning(f"{url} attempt {attempt} failed: {e}")
                        if attempt < DOWNLOAD_ATTEMPTS_PER_SERVER:
                            time.sleep(10 * attempt)
            raise RuntimeError(f"All Overpass servers failed. Last error: {last_err}")
        finally:
            ox.settings.overpass_url = default_url

    def get_graph_with_buffer(self):
        """
        Retrieve a network graph from OpenStreetMap for a buffered AOI.

        The AOI is buffered in meters (EPSG:3857), reprojected back to EPSG:4326,
        and the bounding box of the buffered area is used to download the network graph.
        Saves nodes and edges as GeoPackage files in the 'Intermediate' output folder.

        Returns
        -------
        networkx.MultiDiGraph
            The OSMnx network graph for the buffered AOI.
        """

        logger.info('Retrieving network graph for given AOI')
        inter_dir = os.path.join(self.outfolder, 'Intermediate')
        os.makedirs(inter_dir, exist_ok=True)

        aoi_3857 = self.aoi_gdf.copy()
        aoi_3857 = aoi_3857.to_crs(epsg=3857)

        buffered_aoi = aoi_3857.geometry.buffer(5000)

        extended_boundary = buffered_aoi.geometry.iloc[0]
        gseries_3857 = gpd.GeoSeries([extended_boundary], crs='EPSG:3857')
        gseries_4326 = gseries_3857.to_crs(epsg=4326)
        extended_boundary_4326 = gseries_4326.iloc[0]
        minx, miny, maxx, maxy = extended_boundary_4326.bounds

        self.init_graph = self._download_graph([minx,miny,maxx,maxy])
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.init_graph)

        gdf_nodes.to_file(os.path.join(inter_dir,'init_nodes.gpkg'))
        gdf_edges.to_file(os.path.join(inter_dir,'init_edges.gpkg'))

        return self.init_graph

    def compute_end_points(self):
        """
        Compute intersection points between the AOI boundary and the network edges.

        Converts the AOI polygons to boundaries and finds intersection points with the network edges.
        Adds longitude, latitude, and highway type to each intersection point.

        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with columns: geometry, lon, lat, highway, osmid, ref.
            Each row represents an intersection point.
        """
        logger.info('Computing end points based on roads intersection with AOI')

        # Convert graph edges to GeoDataFrame
        edges = ox.graph_to_gdfs(self.init_graph, nodes=False, edges=True)

        aoi_lines = self.aoi_gdf.copy()

        aoi_lines['geometry'] = aoi_lines['geometry'].boundary
        # Find intersections with original AOI
        edges = edges.to_crs(aoi_lines.crs)
        intersected = gpd.overlay(edges, aoi_lines, how='intersection',keep_geom_type=False)
        intersected_points = intersected.loc[intersected.geom_type == 'Point'].copy()
        intersected_points['lon'] = intersected_points.geometry.x
        intersected_points['lat'] = intersected_points.geometry.y
        intersected_points['highway'] = intersected_points['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
        logger.info(f"Found a total of {len(intersected_points)} points.")

        return intersected_points[['geometry', 'lon', 'lat', 'highway', 'osmid','ref']]

    def retrieve_by_hierarchy(self, df):
        """
        Filter and prioritize end points by road hierarchy.

        Assigns a hierarchy score to each end point based on OSM 'highway' type,
        sorts by hierarchy descending, drops duplicate geometries, and filters out
        lower hierarchy roads (hierarchy <= 2, lower than tertiary).

        Parameters
        ----------
        df : pandas.DataFrame or geopandas.GeoDataFrame
            DataFrame containing at least a 'highway' and 'geometry' column.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            Filtered DataFrame with prioritized end points.
        """

        logger.info('Applying hierarchy on end points and final selection')
        # 1. Assign hierarchy first
        hierarchy_map = {
            'motorway': 7,
            'trunk': 6,
            'primary': 5,
            'secondary': 4,
            'tertiary': 3,
            'residential': 2,
            'unclassified': 1
        }
        df['highway'] = df['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
        df['hierarchy'] = df['highway'].map(hierarchy_map).fillna(0).astype(int)
        # 2. Sort by hierarchy descending
        df = df.sort_values(by='hierarchy', ascending=False)
        # 3. Drop duplicates, keeping the highest hierarchy
        df = df.drop_duplicates(subset=['geometry'])
        # df = df.drop_duplicates(subset=['ref'])
        df = df.loc[df['hierarchy'] > 2]

        logger.info(f"Number of remaining end points after applying filter by hierarchy: {len(df)}")
        return df

    def compute_routes(self,end_points:gpd.GeoDataFrame):
        """
        Calculate shortest routes from the start point to each end point.

        For each end point, finds the nearest network node and computes the shortest path
        from the start node. Routes are dissolved by route_id, duplicates are dropped,
        and geometries are clipped to the AOI.

        Parameters
        ----------
        end_points : geopandas.GeoDataFrame
            DataFrame of end points with 'lon' and 'lat' columns.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame of route geometries clipped to the AOI.
        """

        logger.info('Calculating routes from start point to each end point')

        # Find the nearest node to the start point
        start_node = ox.nearest_nodes(self.init_graph, X=self.start_point.x, Y=self.start_point.y)
        routes = gpd.GeoDataFrame(columns=['geometry'])

        for idx, row in end_points.iterrows():
            # Find the nearest node to the end point
            end_node = ox.nearest_nodes(self.init_graph, X=row['lon'], Y=row['lat'])
            node_attrs = self.init_graph.nodes[end_node]
            lon = node_attrs['x']
            lat = node_attrs['y']
            logger.info(f"For end point at {row['geometry']}, nearest node found at {lon,lat}")

            try:
                # Compute shortest path (list of node IDs)
                route = ox.routing.shortest_path(self.init_graph, start_node, end_node)
                route_gdf = ox.routing.route_to_gdf(self.init_graph, route)
                route_gdf['route_id'] = idx
                routes = pd.concat([routes, route_gdf])
            except Exception as e:
                # If no path found or error, skip this endpoint
                logger.info(f"Error computing route from {self.start_point} to {row['geometry']}: {e}")
                continue
        # Create GeoDataFrame of routes
        # routes_gdf = gpd.GeoDataFrame(geometry=routes, crs=self.aoi_gdf.crs)
        routes = routes.dissolve(by='route_id')
        routes = routes.drop_duplicates(subset=['geometry'])

        routes = gpd.overlay(routes, self.aoi_gdf, how='intersection')
        routes['geometry'] = routes['geometry'].line_merge()
        routes = routes[routes.geometry.type == 'LineString']
        routes = routes.reset_index(drop=True)
        if 'ID' in routes.columns:
            routes = routes.drop(columns=['ID', 'AOI'])
        routes['id'] = routes.index + 1
        logger.info(f"Finished calculation of routes. Final number of routes: {len(routes)}.")
        return routes

    def points_at_thirds_with_direction(self, lines_gdf):
        """
        Compute points at one-third and two-thirds along each route, with direction.

        For each route line, calculates points at 1/3 and 2/3 of its length, estimates
        the local direction (in degrees and as a cardinal direction), and returns these
        as a GeoDataFrame.

        Parameters
        ----------
        lines_gdf : geopandas.GeoDataFrame
            GeoDataFrame of route LineStrings.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with points at 1/3 and 2/3 of each route, including direction
            in degrees ('Graph_Rot') and as a cardinal direction ('Direction').
        """
        logger.info('Calculating direction points for each route')
        points = []
        for idx, row in lines_gdf.iterrows():
            line = row.geometry
            # if not isinstance(line, LineString):
            #     continue
            # Points at 1/3, 2/3, and end of the line
            distances = [line.length / 3, 2 * line.length / 3]
            for dist in distances:
                # Point at this distance
                point = line.interpolate(dist)
                # Small delta for direction calculation
                delta = 0.000001 * line.length
                before_dist = dist - delta
                after_dist = dist + delta
                point_before = line.interpolate(before_dist)
                point_after = line.interpolate(after_dist)
                dx = point_after.x - point_before.x
                dy = point_after.y - point_before.y
                angle_rad = math.atan2(dy, dx)
                angle_deg = int((math.degrees(angle_rad) - 90) % 360)
                points.append({
                    'geometry': point,
                    'direction': angle_deg
                })
        points_gdf = gpd.GeoDataFrame(points, geometry='geometry', crs=lines_gdf.crs)
        points_gdf = points_gdf.rename(columns={'direction': 'Graph_Rot'})
        points_gdf['Graph_Rot'] = points_gdf['Graph_Rot'].astype(int)

        def map_direction(angle):
            angle = angle % 360
            if angle >= 333 or angle <= 27:
                return 'N'
            elif 28 <= angle <= 72:
                return 'NW'
            elif 73 <= angle <= 117:
                return 'W'
            elif 118 <= angle <= 162:
                return 'SW'
            elif 163 <= angle <= 207:
                return 'S'
            elif 208 <= angle <= 252:
                return 'SE'
            elif 253 <= angle <= 297:
                return 'E'
            elif 298 <= angle <= 332:
                return 'NE'
            else:
                return None  # Optional: for unexpected values

        points_gdf['Direction'] = points_gdf['Graph_Rot'].apply(map_direction)
        points_gdf['id'] = points_gdf.index + 1

        logger.info("Completed calculation of direction points.")

        return points_gdf