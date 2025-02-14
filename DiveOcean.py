#! /usr/bin/env python
# -*- python-fmt -*-

## Copyright (c) 2023, 2024  University of Washington.
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

"""Plots Ocean Velocity vs Depth"""

# TODO: This can be removed as of python 3.11
from __future__ import annotations

import typing

import numpy as np
import plotly.graph_objects
import scipy

if typing.TYPE_CHECKING:
    import BaseOpts

import PlotUtils
import PlotUtilsPlotly
from BaseLog import log_info, log_warning
from Plotting import plotdivesingle


@plotdivesingle
def plot_ocean_velocity(
    base_opts: BaseOpts.BaseOptions,
    dive_nc_file: scipy.io._netcdf.netcdf_file,
    generate_plots=True,
    dbcon=None,
) -> tuple[list, list]:
    """Plots ocean velocity"""

    ret_plots = []
    ret_figs = []

    if "ad2cp_inv_glider_uocn" not in dive_nc_file.variables or not generate_plots:
        return (ret_plots, ret_figs)

    try:
        uocn_var = dive_nc_file.variables["ad2cp_inv_profile_uocn"]
        uocn = uocn_var[:]
        uocn[uocn == uocn_var._FillValue] = np.nan
        vocn_var = dive_nc_file.variables["ad2cp_inv_profile_vocn"]
        vocn = vocn_var[:]
        vocn[vocn == vocn_var._FillValue] = np.nan

        deepest_i = (
            max(np.nonzero(np.logical_or(np.logical_not(np.isnan(vocn)), np.logical_not(np.isnan(vocn))))[0]) + 1
        )

        depth = dive_nc_file.variables["ad2cp_inv_profile_depth"][:]
    except KeyError as e:
        log_warning(f"Could not find variable {str(e)} - skipping plot_ts")
        return (ret_plots, ret_figs)
    except Exception:
        log_info("Could not load nc varibles for plot_ts - skipping", "exc")
        return (ret_plots, ret_figs)

    fig = plotly.subplots.make_subplots(
        rows=1, cols=2, subplot_titles=("Eastward Ocean Velocity", "Northward Ocean Velocity")
    )

    fig.add_trace(
        {
            "x": uocn[:deepest_i, 0],
            "y": depth[:deepest_i],
            "type": "scatter",
            "name": "Eastward Ocean Velocity Dive",
            # "mode": "markers",
            "mode": "lines",
            "marker": {
                # "symbol": "triangle-down",
                "color": "DarkBlue",
                #'line':{'width':1, 'color':'LightSlateGrey'}
            },
            "hovertemplate": "%{x:.3f} m/s<br>%{y:.2f} meters<br><extra></extra>",
        },
        row=1,
        col=1,
    )
    fig.add_trace(
        {
            "x": uocn[:deepest_i, 1],
            "y": depth[:deepest_i],
            "type": "scatter",
            "name": "Eastward Ocean Velocity Climb",
            # "mode": "markers",
            "mode": "lines",
            "marker": {
                # "symbol": "triangle-up",
                "color": "DarkGreen",
                #'line':{'width':1, 'color':'LightSlateGrey'}
            },
            "hovertemplate": "%{x:.3f} m/s<br>%{y:.2f} meters<br><extra></extra>",
        },
        row=1,
        col=1,
    )

    fig.add_trace(
        {
            "x": vocn[:deepest_i, 0],
            "y": depth[:deepest_i],
            "type": "scatter",
            "name": "Northward Ocean Velocity Dive",
            # "mode": "markers",
            "mode": "lines",
            "marker": {
                # "symbol": "triangle-down",
                "color": "DarkMagenta",
                #'line':{'width':1, 'color':'LightSlateGrey'}
            },
            "hovertemplate": "%{x:.3f} m/s<br>%{y:.2f} meters<br><extra></extra>",
        },
        row=1,
        col=2,
    )
    fig.add_trace(
        {
            "x": vocn[:deepest_i, 1],
            "y": depth[:deepest_i],
            "type": "scatter",
            "name": "Northward Ocean Velocity Climb",
            # "mode": "markers",
            "mode": "lines",
            "marker": {
                # "symbol": "triangle-up",
                "color": "DarkRed",
                #'line':{'width':1, 'color':'LightSlateGrey'}
            },
            "hovertemplate": "%{x:.3f} m/s<br>%{y:.2f} meters<br><extra></extra>",
        },
        row=1,
        col=2,
    )

    mission_dive_str = PlotUtils.get_mission_dive(dive_nc_file)
    title_text = f"{mission_dive_str}<br>AD2CP Ocean Velocity vs Depth"

    fig.update_layout(
        {
            "xaxis": {
                "title": "Velocity (m/s)",
                "showgrid": True,
            },
            "yaxis": {
                "title": "Depth (m)",
                "autorange": "reversed",
            },
            "xaxis2": {
                "title": "Velocity (m/s)",
                "showgrid": True,
            },
            "yaxis2": {
                "title": "Depth (m)",
                "autorange": "reversed",
            },
            "title": {
                "text": title_text,
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "y": 0.95,
            },
            "margin": {
                "t": 100,
            },
        }
    )

    ret_figs.append(fig)
    ret_plots.extend(
        PlotUtilsPlotly.write_output_files(
            base_opts,
            f"dv{dive_nc_file.dive_number:04d}_ocean_velocity",
            fig,
        )
    )

    return (ret_figs, ret_plots)
