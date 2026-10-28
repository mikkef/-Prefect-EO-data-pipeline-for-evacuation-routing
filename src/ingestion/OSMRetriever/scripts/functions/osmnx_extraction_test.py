import pytest
from unittest.mock import patch, MagicMock
from .osmnx_extraction import OSM_retriever, BBOXError

@pytest.fixture
def retriever():
    return OSM_retriever()

def test_init_with_invalid_bbox_type():
    with patch('loguru.logger.error') as mock_log:
        OSM_retriever(bbox=1234)
        mock_log.assert_called()

def test_init_with_invalid_bbox_format():
    with patch('loguru.logger.error') as mock_log:
        OSM_retriever(bbox="1,2,3")
        mock_log.assert_called()

def test_define_bbox_with_valid_string(retriever):
    retriever.define_bbox(bbox="10,20,30,40")
    assert retriever.bbox == "10,20,30,40"

def test_define_bbox_with_invalid_type(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.define_bbox(bbox=1234)
        mock_log.assert_called()

def test_define_bbox_with_nonexistent_aoi(retriever):
    with patch('os.path.exists', return_value=False), patch('loguru.logger.error') as mock_log:
        retriever.define_bbox(aoi="fake_path.shp")
        mock_log.assert_called()

def test_read_from_file_invalid_type(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.read_from_file(1234)
        mock_log.assert_called()

def test_read_from_file_nonexistent(retriever):
    with patch('os.path.exists', return_value=False), patch('loguru.logger.error') as mock_log:
        retriever.read_from_file("fake.geojson")
        mock_log.assert_called()

def test_get_osm_data_no_bbox(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.get_osm_data()
        mock_log.assert_called()

def test_extract_roads_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_roads()
        mock_log.assert_called()

def test_extract_rail_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_rail()
        mock_log.assert_called()

def test_extract_buildings_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_buildings()
        mock_log.assert_called()

def test_extract_places_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_places()
        mock_log.assert_called()

def test_extract_powergrid_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_powergrid()
        mock_log.assert_called()

def test_extract_poi_no_data(retriever):
    with patch('loguru.logger.error') as mock_log:
        retriever.data = None
        retriever.extract_poi()
        mock_log.assert_called()