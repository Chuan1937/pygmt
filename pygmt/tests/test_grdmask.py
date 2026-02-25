"""
Test pygmt.grdmask.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pygmt import grdmask
from pygmt.enums import GridRegistration, GridType
from pygmt.exceptions import GMTParameterError
from pygmt.helpers import GMTTempFile


@pytest.fixture(scope="module", name="polygon_data")
def fixture_polygon_data():
    """
    Create a simple polygon for testing.
    A triangle polygon covering the region [125, 130, 30, 35].
    """
    return np.array([[125, 30], [130, 30], [130, 35], [125, 30]])


@pytest.fixture(scope="module", name="expected_grid")
def fixture_expected_grid():
    """
    Load the expected grdmask grid result.
    """
    return xr.DataArray(
        data=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        coords={
            "x": [125.0, 126.0, 127.0, 128.0, 129.0, 130.0],
            "y": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        },
        dims=["y", "x"],
    )


def test_grdmask_outgrid(polygon_data, expected_grid):
    """
    Creates a mask grid with an outgrid argument.
    """
    with GMTTempFile(suffix=".nc") as tmpfile:
        result = grdmask(
            data=polygon_data,
            outgrid=tmpfile.name,
            spacing=1,
            region=[125, 130, 30, 35],
        )
        assert result is None  # return value is None
        assert Path(tmpfile.name).stat().st_size > 0  # check that outgrid exists
        temp_grid = xr.load_dataarray(tmpfile.name, engine="gmt", raster_kind="grid")
        # Check grid properties
        assert temp_grid.dims == ("y", "x")
        assert temp_grid.gmt.gtype is GridType.CARTESIAN
        assert temp_grid.gmt.registration is GridRegistration.GRIDLINE
        # Check grid values
        xr.testing.assert_allclose(a=temp_grid, b=expected_grid)


@pytest.mark.benchmark
def test_grdmask_no_outgrid(polygon_data, expected_grid):
    """
    Test grdmask with no set outgrid.
    """
    result = grdmask(data=polygon_data, spacing=1, region=[125, 130, 30, 35])
    # Check grid properties
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y", "x")
    assert result.gmt.gtype is GridType.CARTESIAN
    assert result.gmt.registration is GridRegistration.GRIDLINE
    # Check grid values
    xr.testing.assert_allclose(a=result, b=expected_grid)


def test_grdmask_custom_mask_values(polygon_data):
    """
    Test grdmask with custom mask_values.
    """
    result = grdmask(
        data=polygon_data,
        spacing=1,
        region=[125, 130, 30, 35],
        mask_values=[10, 20, 30],  # outside, edge, inside
    )
    assert isinstance(result, xr.DataArray)
    # Check that the grid has the right dimensions
    assert result.shape == (6, 6)
    # Check that we have values in the expected range
    assert result.values.max() <= 30.0
    assert result.values.min() >= 0.0


def test_grdmask_fails():
    """
    Check that grdmask fails correctly when region and spacing are not given.
    """
    with pytest.raises(GMTParameterError):
        grdmask(data=np.array([[0, 0], [1, 1], [1, 0], [0, 0]]))
