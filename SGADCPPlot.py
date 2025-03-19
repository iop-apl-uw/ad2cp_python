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
SGADCP.py - main entry point for stand alone processing of the Seaglider
ADCP data
"""

import os
import pathlib
import pdb
import sys
import time
import traceback
import warnings

import cmocean
import gsw
import numpy as np
import xarray as xr
from plotly.subplots import make_subplots

import ADCPOpts
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_error, log_warning

DEBUG_PDB = False

# TODO - make options
# IOP Standard Figure size
std_width = 1058
std_height = 894
std_scale = 1.0


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


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


def write_output_files(adcp_opts, base_file_name, fig):
    """
    Helper routine to output various file formats - .png and .div all the time
    and standalone .html and .svg based on conf file settings

    Input:
        adcp_opts - all options
        base_file_name - file name base for the output file names (i.e. no extension)
        fig - plotly figure object
    Returns:
        List of fully qualified filenames that have been generated.
    """
    std_config_dict = {
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["v1hovermode", "toggleSpikelines"],
    }

    if not adcp_opts.plot_directory:
        log_warning("plot_directory not specified - bailing out")
        return []

    base_file_name = os.path.join(adcp_opts.plot_directory, base_file_name)

    ret_list = []

    # if plot_opts full_html
    output_name = base_file_name + ".html"
    fig.write_html(
        file=output_name,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
        validate=True,
        config=std_config_dict,
        include_mathjax="cdn",
    )
    ret_list.append(output_name)

    def save_img_file(output_fmt):
        output_name = base_file_name + "." + output_fmt
        # No return code
        # TODO - for kelido 0.2.1 and python 3.10 (and later) we get this warning:
        #   File "/Users/gbs/.pyenv/versions/3.10.7/lib/python3.10/threading.py", line 1224, in setDaemon
        #   warnings.warn('setDaemon() is deprecated, set the daemon attribute instead',
        #   DeprecationWarning: setDaemon() is deprecated, set the daemon attribute instead
        #
        # Remove when kelido is updated
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fig.write_image(
                output_name,
                format=output_fmt,
                width=std_width,
                height=std_height,
                scale=std_scale,
                validate=True,
                engine="kaleido",
            )
        return output_name

    ret_list.append(save_img_file("webp"))

    return ret_list


#
# Plot routines
#


def PlotOceanVelocity(ncf_name, ds, adcp_opts):
    # Preliminaries
    try:
        vocn = ds.variables["ad2cp_inv_profile_vocn"][:].filled(fill_value=np.nan)
        uocn = ds.variables["ad2cp_inv_profile_uocn"][:].filled(fill_value=np.nan)
        dive_num = ds.variables["ad2cp_inv_profile_dive"][:]
        # profile_time = ds.variables["ad2cp_inv_profile_time"][:]
        depth = ds.variables["ad2cp_inv_profile_depth"][:]
        profile_temperature = ds.variables["ad2cp_profile_temperature"][:]
        profile_salinity = ds.variables["ad2cp_profile_salinity"][:]
        latitude = ds.variables["ad2cp_inv_profile_latitude"][:]
        longitude = ds.variables["ad2cp_inv_profile_longitude"][:]
        mission_summary_str = ds.getncattr("summary")
    except KeyError as e:
        log_warning(f"Could not load {e}")
        return
    except Exception:
        log_error("Problems with load", "exc")
        return

    depth_i = np.logical_and(depth >= adcp_opts.min_plot_depth, depth <= adcp_opts.max_plot_depth)

    profile_latitude = np.tile(np.atleast_2d(latitude).T, np.shape(profile_temperature)[0]).T
    profile_longitude = np.tile(np.atleast_2d(longitude).T, np.shape(profile_temperature)[0]).T
    profile_depth = np.tile(np.atleast_2d(depth).T, np.shape(profile_temperature)[1])

    profile_pressure = gsw.p_from_z(-profile_depth, profile_latitude)
    profile_salinity_absolute = gsw.SA_from_SP(profile_salinity, profile_pressure, profile_longitude, profile_latitude)
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
        temp = ADCPUtils.isoSurface(pd_ds, dc, "depth")
        temp[temp < adcp_opts.min_plot_depth] = np.nan
        temp[temp > adcp_opts.max_plot_depth] = np.nan
        if np.logical_not(np.isnan(temp)).size != 0:
            contours[dc] = temp

    contour_dive_num = np.copy(dive_num).astype(np.float32)
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

    zmax = np.nanmax((np.abs(uocn), np.abs(vocn)))
    # zmax = 1.0
    zmin = -zmax

    # Notes - specifying "contour" instead of "heatmap" yields bad coloring - the center of the colorscale is not zero
    #
    fig.add_trace(
        {
            "x": dive_num,
            "y": depth[depth_i],
            "z": uocn[depth_i, :],
            "type": "heatmap",
            "coloraxis": "coloraxis",
            "hovertemplate": f"Dive %{{x:.0f}}<br>Depth %{{y}} meters<br>%{{z{format_spec}}}{unit_tag}<extra></extra>",
        },
        row=1,
        col=1,
    )

    fig.add_trace(
        {
            "x": dive_num,
            "y": depth[depth_i],
            "z": vocn[depth_i, :],
            "type": "heatmap",
            "coloraxis": "coloraxis",
            "hovertemplate": f"Dive %{{x:.0f}}<br>Depth %{{y}} meters<br>%{{z{format_spec}}}{unit_tag}<extra></extra>",
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
                    "legendgroup": "isopycnals_group",
                    "showlegend": f_show_legend,
                    "hovertemplate": f"{label}<br>Dive %{{x:.0f}}<br>Depth %{{y:.1f}} meters<extra></extra>",
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
                "colorscale": cmocean_to_plotly("balance", 100),
                "colorbar": {
                    # "title": "m s-1",
                    "title": {
                        "text": "m s-1",
                        # "side": "right",
                        "side": "top",
                    },
                    "len": 0.9,
                },
                # "colorscale": "balance",
                # "colorscale": "RdBu",
                "cmin": zmin,
                "cmax": zmax,
                # "cmin": -0.5,
                # "cmax": 0.5,
                "cmid": 0.0,
            },
            "margin": {
                "t": 120,
            },
        }
    )

    # fig.show()
    outs = write_output_files(adcp_opts, pathlib.Path(ncf_name.stem + "_ocean_velocity"), fig)

    return (
        [fig],
        outs,
    )


def GeneratePlots(ncf_name, adcp_opts) -> None:
    ds = ADCPUtils.open_netcdf_file(ncf_name, mask_results=True)
    if ds is None:
        return

    if adcp_opts.min_plot_depth > adcp_opts.max_plot_depth:
        log_error(f"min_plot_depth:{adcp_opts.min_plot_depth:.2f} > max_plot_depth:{adcp_opts.max_plot_depth:.2f}")
        return

    for plot_func in (PlotOceanVelocity,):
        try:
            plot_func(ncf_name, ds, adcp_opts)
        except Exception:
            DEBUG_PDB_F()
            log_error(f"Problem plotting  {plot_func}", "exc")
            continue

    ds.close()


def main(cmdline_args: list[str] = sys.argv[1:]) -> int:
    """Command line driver Seaglider ADCP plotting

    Returns:
        0 for success
        Non-zero for critical problems.

    Raises:
        Any exceptions raised are considered critical errors and not expected
    """

    # NOTE: Minimum options here - debug level and name for yaml config file

    # Get options
    adcp_opts = ADCPOpts.ADCPOptions(
        "Command line driver for Seaglider ADCPx plotting",
        calling_module=pathlib.Path(__file__).stem,
        cmdline_args=cmdline_args,
    )
    if adcp_opts is None:
        return

    global DEBUG_PDB
    DEBUG_PDB = adcp_opts.debug_pdb

    # Initialize log
    ADCPLogger(adcp_opts, include_time=True)

    if ADCPUtils.check_versions():
        return 1

    if adcp_opts.plot_directory is None:
        adcp_opts.plot_directory = adcp_opts.ncf_filename.parent.joinpath("plots")

    if ADCPUtils.SetupPlotDirectory(adcp_opts):
        return 1

    GeneratePlots(adcp_opts.ncf_filename, adcp_opts)

    return 0


if __name__ == "__main__":
    # Force to be in UTC
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        ret_val = main()
    except Exception:
        DEBUG_PDB_F()
        log_critical("Unhandled exception in main -- exiting", "exc")

    sys.exit(ret_val)
