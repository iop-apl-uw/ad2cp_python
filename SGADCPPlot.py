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
import numpy as np
from plotly.subplots import make_subplots
import plotly

import ADCPOpts
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_error, log_warning

DEBUG_PDB = True

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
        profile_time = ds.variables["ad2cp_inv_profile_time"][:]
        depth = ds.variables["ad2cp_inv_profile_depth"][:]
    except KeyError as e:
        log_warning(f"Could not load {e}")
        return
    except Exception:
        log_error("Problems with load", "exc")
        return

    depth_i = np.logical_and(depth >= adcp_opts.min_plot_depth, depth <= adcp_opts.max_plot_depth)

    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Eastward Ocean Velocity", "Northward Ocean Velocity"),
        vertical_spacing=0.1,
        # TODO - helps with synchronizing the zoom of the two windows, but hides the x-axis labels in the upper plot
        # Maybe https://community.plotly.com/t/subplots-with-shared-x-axes-but-show-x-axis-for-each-plot/34800 to fix
        shared_xaxes="all",
        shared_yaxes="all",
    )

    contours = {
        "coloring": "heatmap",
        "showlabels": True,
        "labelfont": {"family": "Raleway", "size": 12, "color": "white"},
    }

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

    # TODO - get this correct
    # mission_dive_str = PlotUtils.get_mission_dive(dive_nc_file)
    mission_dive_str = ""
    title_text = f"{mission_dive_str}<br>AD2CP Ocean Velocity vs Depth"

    # pdb.set_trace()

    fig.update_layout(
        {
            "xaxis": {
                # "title": "Dive Number",
                "showgrid": True,
            },
            "yaxis": {
                "title": "Depth (m)",
                "autorange": "reversed",
            },
            "xaxis2": {
                "title": "DiveNumber",
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
            "coloraxis": {
                "colorscale": cmocean_to_plotly("balance", 100),
                "colorbar": {
                    # "title": "m s-1",
                    "title": {
                        "text": "m s-1",
                        # "side": "right",
                        "side": "top",
                    }
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
                "t": 100,
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
        log_error(f"min_plot_depth:{min_plot_depth:.2f} > max_plot_depth:{max_plot_depth:.2f}")
        return

    for plot_func in (PlotOceanVelocity,):
        try:
            plot_func(ncf_name, ds, adcp_opts)
        except Exception:
            DEBUG_PDB_F()
            log_error(f"Problem plotting  {plot_func}", "exc")
            continue

    ds.close()


def main() -> int:
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
        "Command line driver for Seaglider ADCPx plotting", calling_module=pathlib.Path(__file__).stem
    )

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