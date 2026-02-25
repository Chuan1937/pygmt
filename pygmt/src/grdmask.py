"""
grdmask - Create mask grid from polygons or point coverage.
"""

from collections.abc import Sequence
from typing import Literal

import xarray as xr
from pygmt._typing import PathLike
from pygmt.alias import Alias, AliasSystem
from pygmt.clib import Session
from pygmt.exceptions import GMTParameterError
from pygmt.helpers import build_arg_list, fmt_docstring

__doctest_skip__ = ["grdmask"]


@fmt_docstring
def grdmask(
    data,
    outgrid: PathLike | None = None,
    spacing: Sequence[float | str] | None = None,
    region: Sequence[float | str] | str | None = None,
    mask_values: Sequence[float] | None = None,
    verbose: Literal["quiet", "error", "warning", "timing", "info", "compat", "debug"]
    | bool = False,
    **kwargs,
) -> xr.DataArray | None:
    """
    Create mask grid from polygons or point coverage.

    Reads one or more files (or standard input) containing polygon or data point
    coordinates, and creates a binary grid file where nodes that fall inside, on the
    edge, or outside the polygons (or within the search radius from data points) are
    assigned values based on ``mask_values``.

    The mask grid can be used to mask out specific regions in other grids using
    :func:`pygmt.grdmath` or similar tools. For masking based on coastline features,
    consider using :func:`pygmt.grdlandmask` instead.

    Full GMT docs at :gmt-docs:`grdmask.html`.

    $aliases
       - G = outgrid
       - I = spacing
       - N = mask_values
       - R = region
       - V = verbose

    Parameters
    ----------
    data
        Pass in either a file name, :class:`pandas.DataFrame`, :class:`numpy.ndarray`,
        or a list of file names containing the polygon(s) or data points. Input can be:

        - **Polygon mode**: One or more files containing closed polygon coordinates
        - **Point coverage mode**: Data points (used with ``search_radius`` parameter)
    $outgrid
    $spacing
    mask_values : list of float, optional
        Set the values that will be assigned to nodes. Provide three values in the form
        [*outside*, *edge*, *inside*]. Default is ``[0, 0, 1]``, meaning nodes outside
        and on the edge are set to 0, and nodes inside are set to 1.

        Values can be any number, or one of ``None``, ``"NaN"``, and ``np.nan`` for
        setting nodes to NaN.
    $region
    $verbose

    Returns
    -------
    ret
        Return type depends on whether the ``outgrid`` parameter is set:

        - :class:`xarray.DataArray` if ``outgrid`` is not set
        - ``None`` if ``outgrid`` is set (grid output will be stored in the file set by
          ``outgrid``)

    Example
    -------
    >>> import pygmt
    >>> import numpy as np
    >>> # Create a simple polygon as a triangle
    >>> polygon = np.array([[125, 30], [130, 30], [130, 35], [125, 30]])
    >>> # Create a mask grid with 1 arc-degree spacing
    >>> mask = pygmt.grdmask(data=polygon, spacing=1, region=[125, 130, 30, 35])
    """
    if spacing is None or region is None:
        raise GMTParameterError(required=["region", "spacing"])

    aliasdict = AliasSystem(
        I=Alias(spacing, name="spacing", sep="/", size=2),
        N=Alias(mask_values, name="mask_values", sep="/", size=3),
    ).add_common(
        R=region,
        V=verbose,
    )
    aliasdict.merge(kwargs)

    with Session() as lib:
        with (
            lib.virtualfile_in(check_kind="vector", data=data) as vintbl,
            lib.virtualfile_out(kind="grid", fname=outgrid) as voutgrd,
        ):
            aliasdict["G"] = voutgrd
            lib.call_module(
                module="grdmask",
                args=build_arg_list(aliasdict, infile=vintbl),
            )
            return lib.virtualfile_to_raster(vfname=voutgrd, outgrid=outgrid)
