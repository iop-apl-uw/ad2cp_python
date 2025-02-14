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

"""Plots Ocean Velocity profiles"""

# TODO: This can be removed as of python 3.11
from __future__ import annotations

import os
import pathlib
import sys
import typing

import gsw
import numpy as np
import xarray as xr
import yaml
from plotly.subplots import make_subplots

# pylint: disable=wrong-import-position
if typing.TYPE_CHECKING:
    import BaseOpts

import PlotUtilsPlotly
import Utils
import Utils2
from BaseLog import log_error, log_info, log_warning
from Plotting import plotmissionsingle

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import ADCPPlotUtils


def getValue(x, v, sk, vk, fallback):
    if v in x["sections"][sk]:
        y = x["sections"][sk][v]
    elif v in x["variables"][vk]:
        y = x["variables"][vk][v]
    elif v in x["defaults"]:
        y = x["defaults"][v]
    else:
        y = fallback

    return y


@plotmissionsingle
def mission_oceanvelocityprofile(
    base_opts: BaseOpts.BaseOptions,
    mission_str: list,
    dive=None,
    generate_plots=True,
    dbcon=None,
) -> tuple[list, list]:
    ret_plots = []
    ret_figs = []

    if not generate_plots:
        return (ret_figs, ret_plots)

    # CONSIDER - moving all of this to a common routine (with SGADCPPlot)
    # so the section handling is common.  Pros - SGAPCPPlot get section
    # handing.  Cons - eventually might get complicated as a general template
    # for plots for stand alone users

    # Process the section file
    section_file_name = pathlib.Path(base_opts.mission_dir).joinpath("sections.yml")
    if not section_file_name.exists():
        return (ret_figs, ret_plots)

    try:
        with open(section_file_name, "r") as f:
            section_dict = yaml.safe_load(f.read())
    except Exception:
        log_error(f"Problem processing {section_file_name}", "exc")
        return (ret_figs, ret_plots)

    if "variables" not in section_dict or len(section_dict["variables"]) == 0:
        log_error(f"No 'variables' key found in {section_file_name} - not plotting")
        return (ret_figs, ret_plots)

    if "sections" not in section_dict or len(section_dict["sections"]) == 0:
        log_error(f"No 'sections' key found in {section_file_name} - not plotting")
        return (ret_figs, ret_plots)

    if dbcon is None:
        conn = Utils.open_mission_database(base_opts, ro=True)
        if not conn:
            log_error("Could not open mission database")
            return ([], [])

        log_info("mission_profiles db opened (ro)")
    else:
        conn = dbcon

    try:
        cur = conn.cursor()
        cur.execute("SELECT dive FROM dives ORDER BY dive DESC LIMIT 1;")
        res = cur.fetchone()
        cur.close()
        if res:
            latest = res[0]
        else:
            log_warning("No dives found")
            if dbcon is None:
                conn.close()
                log_info("mission_profiles db closed")
            return ([], [])
    except Exception:
        log_error("Could not fetch data", "exc")
        if dbcon is None:
            conn.close()
            log_info("mission_profiles db closed")
        return ([], [])

    if dbcon is None:
        conn.close()
        log_info("mission_profiles db closed")

    for adcp_varn in ("ad2cp_inv_profile_vocn", "ad2cp_inv_profile_uocn"):
        if adcp_varn in section_dict["variables"]:
            break
    else:
        return (ret_figs, ret_plots)

    ncname = Utils2.get_mission_timeseries_name(base_opts, basename="adcp_profile_timeseries")

    try:
        ds = Utils.open_netcdf_file(ncname, "r", mask_results=True)
    except Exception:
        log_error(f"Unable to open {ncname}", "exc")
        return (ret_figs, ret_plots)

    try:
        latest = min(ds.variables["ad2cp_inv_profile_dive"][-1], latest)
    except KeyError as e:
        log_warning(f"Could not load {e}")
        return (ret_figs, ret_plots)
    except Exception:
        log_error("Problems with load", "exc")
        return (ret_figs, ret_plots)

    for sk in section_dict["sections"]:
        start = getValue(section_dict, "start", sk, adcp_varn, 1)
        # TODO? step = getValue(section_dict, "step", sk, adcp_varn, 1)
        stop = getValue(section_dict, "stop", sk, adcp_varn, -1)
        # TODO flip = getValue(section_dict, "flip", sk, adcp_varn, False)
        cmap = getValue(section_dict, "colormap", sk, adcp_varn, "balance")
        # These may not make sense to do
        # TODO? zmin = getValue(section_dict, "min", sk, adcp_varn, None)
        # TODO? zmax = getValue(section_dict, "max", sk, adcp_varn, None)
        # TODO? fill = getValue(section_dict, "fill", sk, adcp_varn, False)
        top = getValue(section_dict, "top", sk, adcp_varn, 0)
        bott = getValue(section_dict, "bottom", sk, adcp_varn, 990)

        if stop == -1 or stop >= latest:
            stop = latest
            force = True
        else:
            force = False

        fname = f"sg_ad2cp_inv_profile_section_{sk}"
        fullname = pathlib.Path(base_opts.mission_dir).joinpath(f"plots/{fname}.webp")
        if fullname.exists() and section_file_name.stat().st_mtime < fullname.stat().st_mtime and not force:
            continue

        try:
            dive_num = ds.variables["ad2cp_inv_profile_dive"][:]
        except KeyError as e:
            log_warning(f"Could not load {e}")
            return (ret_figs, ret_plots)

        dive_i = np.nonzero(np.logical_and(dive_num >= start, dive_num <= stop))[0]

        # Preliminaries
        try:
            vocn = ds.variables["ad2cp_inv_profile_vocn"][:, dive_i].filled(fill_value=np.nan)
            uocn = ds.variables["ad2cp_inv_profile_uocn"][:, dive_i].filled(fill_value=np.nan)
            dive_num = ds.variables["ad2cp_inv_profile_dive"][dive_i]
            # profile_time = ds.variables["ad2cp_inv_profile_time"][:]
            depth = ds.variables["ad2cp_inv_profile_depth"][:]
            profile_temperature = ds.variables["ad2cp_profile_temperature"][:, dive_i]
            profile_salinity = ds.variables["ad2cp_profile_salinity"][:, dive_i]
            latitude = ds.variables["ad2cp_inv_profile_latitude"][dive_i]
            longitude = ds.variables["ad2cp_inv_profile_longitude"][dive_i]
            mission_summary_str = ds.getncattr("summary")
        except KeyError as e:
            log_warning(f"Could not load {e}")
            return (ret_figs, ret_plots)
        except Exception:
            log_error("Problems with load", "exc")
            return (ret_figs, ret_plots)

        depth_i = np.logical_and(depth >= top, depth <= bott)
        profile_latitude = np.tile(np.atleast_2d(latitude).T, np.shape(profile_temperature)[0]).T
        profile_longitude = np.tile(np.atleast_2d(longitude).T, np.shape(profile_temperature)[0]).T
        profile_depth = np.tile(np.atleast_2d(depth).T, np.shape(profile_temperature)[1])

        profile_pressure = gsw.p_from_z(-profile_depth, profile_latitude)
        profile_salinity_absolute = gsw.SA_from_SP(
            profile_salinity, profile_pressure, profile_longitude, profile_latitude
        )
        profile_dens = np.ma.getdata(
            gsw.pot_rho_t_exact(profile_salinity_absolute, profile_temperature, profile_pressure, 0.0) - 1000.0
        )

        pd_ds = xr.DataArray(profile_dens, coords={"depth": depth, "dive": dive_num})

        density_contours = np.arange(
            np.floor(np.nanmin(profile_dens) * 10.0) / 10.0, np.ceil(np.nanmax(profile_dens) * 10.0) / 10.0, 0.1
        )

        # Isopycnals
        contours = {}
        for dc in density_contours:
            temp = ADCPPlotUtils.isoSurface(pd_ds, dc, "depth")
            temp[temp < top] = np.nan
            temp[temp > bott] = np.nan
            if np.logical_not(np.isnan(temp)).size != 0:
                contours[dc] = temp

        contour_dive_num = np.copy(dive_num).astype(np.float32)

        #
        # TODO - Not sure what this is for - seems to create the wrong plot by "offsetting" contour
        #        May need to address in SGADCPPlot.py
        #
        for ii in range(0, contour_dive_num.size, 2):
            contour_dive_num[ii] += 0.25
            contour_dive_num[ii + 1] += 0.75

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Eastward Ocean Velocity", "Northward Ocean Velocity"),
            vertical_spacing=0.1,
            # Helps with synchronizing the zoom of the two windows, but hides the x-axis labels in the upper plot
            # Fix below from https://community.plotly.com/t/subplots-with-shared-x-axes-but-show-x-axis-for-each-plot/34800
            shared_xaxes="all",
            shared_yaxes="all",
        )

        format_spec = ":.4f"
        unit_tag = "m s-1"

        cm_max = np.nanmax((np.abs(uocn), np.abs(vocn)))
        # cm_max = 1.0
        cm_min = -cm_max

        # Notes - specifying "contour" instead of "heatmap" yields bad coloring
        # the center of the colorscale is not zero

        customdata = np.tile(dive_num.filled().transpose(), (uocn.shape[0], 1))
        fig.add_trace(
            {
                # "x": dive_num,
                "x": contour_dive_num,
                "y": depth[depth_i],
                "z": uocn[depth_i, :],
                # "meta": dive_num.filled().astype(np.float32),
                "customdata": customdata,
                # "meta": contour_dive_num,
                "type": "heatmap",
                "coloraxis": "coloraxis",
                # "hovertemplate": "Dive %{meta:.0f}<br>Depth %{y} meters<br>"
                "hovertemplate": "Dive %{customdata}<br>Depth %{y} meters<br>"
                + f"%{{z{format_spec}}} {unit_tag}<extra></extra>",
            },
            row=1,
            col=1,
        )

        fig.add_trace(
            {
                # "x": dive_num,
                "x": contour_dive_num,
                "y": depth[depth_i],
                "z": vocn[depth_i, :],
                # "meta": dive_num.filled(),
                "customdata": customdata,
                "type": "heatmap",
                "coloraxis": "coloraxis",
                "hovertemplate": "Dive %{customdata}<br>Depth %{y} meters<br>"
                + f"%{{z{format_spec}}}{unit_tag}<extra></extra>",
            },
            row=2,
            col=1,
        )

        # TODO: Add button to toggle isopycnals on/off
        # TODO: Look for lighter grey
        f_show_legend = True
        for kv, contour in contours.items():
            _, d = np.divmod(kv, 1)
            if np.isclose(d, 0.0):
                color = "DarkGrey"
            elif np.isclose(d, 0.5):
                color = "LightGrey"
            else:
                continue
            for jj in range(1, 3):
                label = f"iso {kv:.1f} kg m-3"
                # name = f"{kv:.1f}_{jj}"

                fig.add_trace(
                    {
                        "x": contour_dive_num,
                        "y": contour,
                        "name": "isopycnals<br>0.5 kg m-s/incr",
                        "type": "scatter",
                        "mode": "lines",
                        "marker": {
                            "line": {
                                "width": 1,
                                "color": color,
                            },
                            "color": color,
                        },
                        # "showlegend": False,
                        # "visible": True,
                        # "visible": "legendonly",
                        "customdata": dive_num,
                        "legendgroup": "isopycnals_group",
                        "showlegend": f_show_legend,
                        "hovertemplate": f"{label}<br>Dive %{{customdata}}<br>Depth %{{y:.1f}} meters<extra></extra>",
                    },
                    row=jj,
                    col=1,
                )
                f_show_legend = False

        title_text = f"{mission_summary_str}<br>AD2CP Ocean Velocity vs Depth"

        fig.update_layout(
            {
                "xaxis": {
                    "title": "Dive Number",
                    "showgrid": True,
                    "showticklabels": True,
                },
                "yaxis": {
                    "title": "Depth (m)",
                    "autorange": "reversed",
                },
                "xaxis2": {
                    "title": "DiveNumber",
                    "showgrid": True,
                    "showticklabels": True,
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
                "coloraxis": {
                    "colorscale": ADCPPlotUtils.cmocean_to_plotly(cmap, 100),
                    "colorbar": {
                        # "title": "m s-1",
                        "title": {
                            "text": "m s-1",
                            # "side": "right",
                            "side": "top",
                        },
                        "len": 0.9,
                    },
                    "cmin": cm_min,
                    "cmax": cm_max,
                    # "cmin": -0.5,
                    # "cmax": 0.5,
                    "cmid": 0.0,
                },
                "margin": {
                    "t": 120,
                },
            }
        )

        ret_figs.append(fig)
        ret_plots.extend(
            PlotUtilsPlotly.write_output_files(
                base_opts,
                fname,
                fig,
            )
        )

    return (ret_figs, ret_plots)
