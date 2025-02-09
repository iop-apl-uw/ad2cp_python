#! /usr/bin/env python
# -*- python-fmt -*-
## Copyright (c) 2023, 2024, 2025  University of Washington.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice, this
##    list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##    this list of conditions and the following disclaimer in the documentation
##    and/or other materials provided with the distribution.
##
## 3. Neither the name of the University of Washington nor the names of its
##    contributors may be used to endorse or promote products derived from this
##    software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS
## IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
## GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
## OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
ADCPPlotUtils.py - Utility functions for plotting
"""

import cmocean
import numpy as np
import xarray as xr


def isoSurface(field, target, dim):
    """
    Linearly interpolate a coordinate isosurface where a field
    equals a target

    Parameters
    ----------
    field : xarray DataArray
        The field in which to interpolate the target isosurface
    target : float
        The target isosurface value
    dim : str
        The field dimension to interpolate

    Examples
    --------
    Calculate the depth of an isotherm with a value of 5.5:

    >>> temp = xr.DataArray(
    ...     range(10,0,-1),
    ...     coords={"depth": range(10)}
    ... )
    >>> isoSurface(temp, 5.5, dim="depth")
    <xarray.DataArray ()>
    array(4.5)
    """
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = field.isel(slice0).drop_vars(dim)
    field1 = field.isel(slice1).drop_vars(dim)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(crossing_mask_decr | crossing_mask_incr, 1, np.nan)

    coords0 = crossing_mask * field[dim].isel(slice0).drop_vars(dim)
    coords1 = crossing_mask * field[dim].isel(slice1).drop_vars(dim)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = coords0 + (target - field0) * (coords1 - coords0) / (field1 - field0)

    return iso.max(dim, skipna=True)


def cmocean_to_plotly(cmapname, pl_entries):
    maps = {
        "thermal": cmocean.cm.thermal,
        "haline": cmocean.cm.haline,
        "solar": cmocean.cm.solar,
        "ice": cmocean.cm.ice,
        "gray": cmocean.cm.gray,
        "oxy": cmocean.cm.oxy,
        "deep": cmocean.cm.deep,
        "dense": cmocean.cm.dense,
        "algae": cmocean.cm.algae,
        "matter": cmocean.cm.matter,
        "turbid": cmocean.cm.turbid,
        "speed": cmocean.cm.speed,
        "amp": cmocean.cm.amp,
        "tempo": cmocean.cm.tempo,
        "phase": cmocean.cm.phase,
        "balance": cmocean.cm.balance,
        "delta": cmocean.cm.delta,
        "curl": cmocean.cm.curl,
    }

    cmap = maps.get(cmapname, cmocean.cm.thermal)

    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale
