import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon

from src.steps.processing.Evacuation.app import run_evacuation


class TestRunEvacuation:
    """Simple test class for run_evacuation Click command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_aoi_file(self, temp_dir):
        """Create a mock AOI shapefile."""
        aoi_file = os.path.join(temp_dir, 'aoi.shp')

        # Create a simple polygon GeoDataFrame
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        aoi_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')
        aoi_gdf.to_file(aoi_file)

        return aoi_file

    @patch('src.steps.processing.Evacuation.app.AOIRoutePlanner')
    def test_run_evacuation_success(self, mock_planner_class, temp_dir, mock_aoi_file):
        """Test successful evacuation route processing."""
        # Mock the AOIRoutePlanner and its methods
        mock_planner = MagicMock()
        mock_planner_class.return_value = mock_planner

        # Mock return values
        mock_end_points = MagicMock()
        mock_planner.compute_end_points.return_value = mock_end_points
        mock_planner.retrieve_by_hierarchy.return_value = mock_end_points

        # Mock routes GeoDataFrame
        mock_routes = MagicMock()
        mock_routes.to_crs.return_value = mock_routes
        mock_routes.drop.return_value = mock_routes
        mock_planner.compute_routes.return_value = mock_routes

        # Mock point directions GeoDataFrame
        mock_points = MagicMock()
        mock_points.to_crs.return_value = mock_points
        mock_planner.points_at_thirds_with_direction.return_value = mock_points

        # Call function directly (bypass Click)
        try:
            run_evacuation.callback(
                aoi_path=mock_aoi_file,
                start_point='26.5425057,39.1348286',
                outfolder=temp_dir
            )
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")

        # Assertions
        assert success

        # Verify AOIRoutePlanner was instantiated
        assert mock_planner_class.called

        # Verify main methods were called
        mock_planner.get_graph_with_buffer.assert_called_once()
        mock_planner.compute_end_points.assert_called_once()
        mock_planner.retrieve_by_hierarchy.assert_called_once_with(mock_end_points)
        mock_planner.compute_routes.assert_called_once_with(mock_end_points)
        mock_planner.points_at_thirds_with_direction.assert_called_once_with(mock_routes)

        # Verify to_file was called for both outputs
        assert mock_routes.to_file.called
        assert mock_points.to_file.called

    def test_run_evacuation_invalid_start_point(self, temp_dir, mock_aoi_file):
        """Test with invalid start point format."""
        try:
            run_evacuation.callback(
                aoi_path=mock_aoi_file,
                start_point='invalid_format',
                outfolder=temp_dir
            )
            success = True
        except Exception:
            success = False

        # Should fail with invalid start point
        assert not success
