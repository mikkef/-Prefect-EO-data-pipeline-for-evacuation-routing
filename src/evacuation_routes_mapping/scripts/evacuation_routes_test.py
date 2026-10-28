import pytest
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import pandas as pd

from evacuation_routes import AOIRoutePlanner

@pytest.fixture
def dummy_aoi():
    poly = Polygon([(0,0), (0,1), (1,1), (1,0), (0,0)])
    gdf = gpd.GeoDataFrame({'geometry': [poly]}, crs='EPSG:4326')
    return gdf

@pytest.fixture
def planner(dummy_aoi, tmp_path):
    return AOIRoutePlanner(dummy_aoi, (0.5, 0.5), 'drive', str(tmp_path))

def test_init_sets_attributes(planner, dummy_aoi, tmp_path):
    assert planner.aoi_gdf.equals(dummy_aoi)
    assert planner.start_point.x == 0.5 and planner.start_point.y == 0.5
    assert planner.network_type == 'drive'
    assert planner.outfolder == str(tmp_path)

@patch('osmnx.graph_from_bbox')
@patch('osmnx.graph_to_gdfs')
def test_get_graph_with_buffer_saves_files(mock_graph_to_gdfs, mock_graph_from_bbox, planner, tmp_path):
    mock_graph = MagicMock()
    mock_graph_from_bbox.return_value = mock_graph
    mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0,0)]}, crs='EPSG:4326')
    mock_edges = gpd.GeoDataFrame({'geometry': [LineString([(0,0),(1,1)])]}, crs='EPSG:4326')
    mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
    with patch.object(gpd.GeoDataFrame, 'to_file') as mock_to_file:
        result = planner.get_graph_with_buffer()
        assert result == mock_graph
        assert mock_to_file.call_count == 2

@patch('osmnx.graph_to_gdfs')
def test_compute_end_points_returns_expected_columns(mock_graph_to_gdfs, planner, dummy_aoi):
    # Mock edges as a GeoDataFrame
    edge = LineString([(0,0), (1,1)])
    edges = gpd.GeoDataFrame({'geometry': [edge], 'highway': ['primary'], 'osmid': [1], 'ref': ['A']}, crs='EPSG:4326')
    mock_graph_to_gdfs.return_value = edges
    planner.init_graph = MagicMock()
    # Patch overlay to return intersection points
    point = Point(0.5, 0.5)
    intersected = gpd.GeoDataFrame({'geometry': [point], 'highway': ['primary'], 'osmid': [1], 'ref': ['A']}, crs='EPSG:4326')
    with patch('geopandas.overlay', return_value=intersected):
        result = planner.compute_end_points()
        assert set(result.columns) == {'geometry', 'lon', 'lat', 'highway', 'osmid', 'ref'}
        assert len(result) == 1

def test_retrieve_by_hierarchy_filters_and_sorts(planner):
    df = pd.DataFrame({
        'geometry': [Point(0,0), Point(1,1)],
        'highway': ['primary', 'residential']
    })
    result = planner.retrieve_by_hierarchy(df)
    assert all(result['hierarchy'] > 2)
    assert result.iloc[0]['highway'] == 'primary'

@patch('osmnx.nearest_nodes')
@patch('osmnx.routing.shortest_path')
@patch('osmnx.routing.route_to_gdf')
@patch('geopandas.overlay')
def test_compute_routes_success(mock_overlay, mock_route_to_gdf, mock_shortest_path, mock_nearest_nodes, planner):
    planner.init_graph = MagicMock()
    mock_nearest_nodes.side_effect = [1, 2]
    mock_shortest_path.return_value = [1, 2]
    route_line = LineString([(0,0), (1,1)])
    mock_route_to_gdf.return_value = gpd.GeoDataFrame({'geometry': [route_line], 'route_id': [0]}, crs='EPSG:4326')
    mock_overlay.return_value = gpd.GeoDataFrame({'geometry': [route_line]}, crs='EPSG:4326')
    end_points = gpd.GeoDataFrame({'lon': [1], 'lat': [1], 'geometry': [Point(1,1)]}, crs='EPSG:4326')
    routes = planner.compute_routes(end_points)
    assert 'geometry' in routes.columns
    assert len(routes) == 1

def test_points_at_thirds_with_direction(planner):
    line = LineString([(0,0), (0,3)])
    lines_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs='EPSG:4326')
    points_gdf = planner.points_at_thirds_with_direction(lines_gdf)
    assert 'Graph_Rot' in points_gdf.columns
    assert 'Direction' in points_gdf.columns
    assert len(points_gdf) == 2
    assert all(points_gdf['Direction'].isin(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']))