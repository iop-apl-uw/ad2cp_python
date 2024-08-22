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

import collections
import os
import pathlib
import pdb
import sys
import time
import traceback

import h5py

# import netCDF4
import numpy as np

import ADCP
import ADCPConfig
import ADCPFiles
import ADCPOpts
import ADCPRealtime
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_debug, log_error, log_info, log_warning

DEBUG_PDB = True


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


def main() -> int:
    """Command line driver Seaglider ADCP data processing

    Returns:
        0 for success
        Non-zero for critical problems.

    Raises:
        Any exceptions raised are considered critical errors and not expected
    """

    # NOTE: Minimum options here - debug level and name for yaml config file

    # Get options
    adcp_opts = ADCPOpts.ADCPOptions(
        "Command line driver Seaglider ADCPx data processing", calling_module=pathlib.Path(__file__).stem
    )

    # Initialize log
    ADCPLogger(adcp_opts, include_time=True)

    if ADCPUtils.check_versions():
        return 1

    # Read the config file
    param, weights = ADCPConfig.ProcessConfigFile(adcp_opts.adcp_config_file)
    if not param:
        return 1

    # param.gz = (0:param.dz:1000)';
    # Done in ADCPConfig

    # Mission-long corrections - not used in the 2024 version of the real-time code
    # if exist('dpitch','var')
    # param.dpitch = dpitch;
    # end
    # if exist('dpitch','var')
    # param.droll = droll;
    # end
    # if exist('dpitch','var')
    # param.dheading = dheading;
    # end

    log_debug(param)

    var_t = collections.namedtuple("var_t", ("accessor", "var_meta_name"))

    mission_vars = {
        "time": var_t(lambda x: x.time, "inverse_profile_time"),
        "uocn": var_t(lambda x: x.UVocn.real, "inverse_profile_velocity_north"),
        "vocn": var_t(lambda x: x.UVocn.imag, "inverse_profile_velocity_east"),
        "wocn": var_t(lambda x: x.Wocn, "inverse_profile_velocity_vertical"),
        "uttw_solution": var_t(lambda x: x.UVttw_solution.real, "inverse_profile_glider_velocity_north"),
        "vttw_solution": var_t(lambda x: x.UVttw_solution.imag, "inverse_profile_glider_velocity_east"),
        "wttw_solution": var_t(lambda x: x.Wttw_solution, "inverse_profile_glider_velocity_vertical"),
        "uttw_model": var_t(lambda x: x.UVttw_model.real, "inverse_profile_model_velocity_north"),
        "vttw_model": var_t(lambda x: x.UVttw_model.imag, "inverse_profile_model_velocity_east"),
        "wttw_model": var_t(lambda x: x.Wttw_model, "inverse_profile_model_velocity_vertical"),
        "uverr": var_t(lambda x: x.UVerr, "inverse_profile_velocity_error"),
    }

    # Whole mission accumulators
    depth_grid = None
    mission_dive = []
    mission_lon = []
    mission_lat = []

    mission_accums = {}
    for k in mission_vars:
        mission_accums[k] = []

    output_filename = None

    global_meta = ADCPConfig.LoadGlobalMeta(adcp_opts.global_meta_filename)

    var_meta = ADCPConfig.LoadVarMeta(adcp_opts.var_meta_filename)

    dive_nc_filenames: list[pathlib.Path] = []

    if adcp_opts.ncf_files:
        for ff in adcp_opts.ncf_files:
            dive_nc_filenames.append(adcp_opts.mission_dir / ff)
    else:
        for m in adcp_opts.mission_dir.glob("p[0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc"):
            dive_nc_filenames.append(m)
    dive_nc_filenames = sorted(dive_nc_filenames)

    for ncf_name in dive_nc_filenames:
        log_info(f"Processing {ncf_name}")
        ds = ADCPUtils.open_netcdf_file(ncf_name)
        if ds is None:
            continue

        if not param.sg:
            param.sg = ds.glider

        if not output_filename:
            output_filename = pathlib.Path(adcp_opts.mission_dir).joinpath(
                f"{ADCPUtils.GetMissionStr(ds).replace(' ', '-').replace('_', '-')}-adcp-realtime.nc"
            )

        # Read real-time
        try:
            glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, ncf_name, param)
        except KeyError as e:
            log_warning(f"{e} - skipping")
            continue
        except Exception:
            DEBUG_PDB_F()
            log_error(f"Problem loading data from {ncf_name}", "exc")
            continue
        finally:
            ds.close()

        param.time_limits = np.array((np.min(gps.log_gps_time), np.max(gps.log_gps_time)))

        # Transform velocites to instrument frame
        try:
            ADCPRealtime.TransformToInstrument(adcp_realtime)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem transforming compass data", "exc")
            continue

        # Clean up adcp data
        try:
            ADCP.CleanADCP(adcp_realtime, glider, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem cleaning realtime adcp data", "exc")
            continue

        # From matlab code - caclulated but unused
        # time_diff = np.diff(adcp_realtime.time)
        # dt = np.max([15.0, np.median(time_diff[time_diff > 2.0])])

        # Perfrom inverse
        try:
            D, profile, variables_for_plot, inverse_tmp = ADCP.Inverse(
                adcp_realtime, gps, glider, weights, param, {} if adcp_opts.save_details else None
            )
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem performing inverse calculation", "exc")
            continue

        # Record the ouput into the accumulators

        mission_dive.append(glider.dive)
        mission_dive.append(glider.dive)

        imax = np.argmax(glider.ctd_depth)
        mission_lon.append(np.nanmean(glider.longitude[: imax + 1]))
        mission_lon.append(np.nanmean(glider.longitude[imax:]))
        mission_lat.append(np.nanmean(glider.latitude[: imax + 1]))
        mission_lat.append(np.nanmean(glider.latitude[imax:]))

        for k, v in mission_vars.items():
            mission_accums[k].append(v.accessor(profile))

        if depth_grid is None:
            depth_grid = profile.z

        # Debug save out - for comparision with other processing
        # TODO - need a better name - match with input file
        if adcp_opts.save_details:
            with h5py.File("test.hdf5", "w") as hdf:
                adcp_realtime.save_to_hdf5("adcp_realtime", hdf)
                glider.save_to_hdf5("glider", hdf)
                gps.save_to_hdf5("gps", hdf)
                D.save_to_hdf5("D", hdf)
                profile.save_to_hdf5("profile", hdf)
                grp = hdf.create_group("inverse_tmp")
                for k, v in inverse_tmp.items():
                    grp.create_dataset(k, data=v)

    # Output results

    # Convert accumulator lists into arrays
    mission_var_arrs = {}
    non_nans = []
    for k in mission_vars:
        mission_var_arrs[k] = np.hstack(mission_accums[k])
        non_nans.append(np.logical_not(np.isnan(mission_var_arrs[k])))

    # Find the deepest data
    deepest_i = max(np.nonzero(np.logical_or.reduce(non_nans))[0]) + 1

    # TODO - matlab code applies these edge cases - appropriate to do?

    # % NaN the edges
    # adcp_grid.time(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVocn(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVttw_solution(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVttw_model(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVerr(i0z,n_dive*2-1:n_dive*2) = NaN;

    # adcp_grid.Wocn(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.Wttw_solution(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.Wttw_model(i0z,n_dive*2-1:n_dive*2) = NaN;

    # Associate variables with meta data and trim
    # data to the deepest observation
    ad2cp_variable_mapping = {
        "inverse_profile_depth": depth_grid[:deepest_i],
        "inverse_profile_dive": np.hstack(mission_dive),
        "inverse_profile_longitude": np.hstack(mission_lon),
        "inverse_profile_latitude": np.hstack(mission_lat),
    }
    for k, v in mission_vars.items():
        ad2cp_variable_mapping[v.var_meta_name] = mission_var_arrs[k][:deepest_i, :]

    dso = ADCPUtils.open_netcdf_file(output_filename, "w")
    if dso is None:
        log_error(f"Could not open {output_filename}")
        return 1

    try:
        ADCPUtils.CreateNCVars(dso, ad2cp_variable_mapping, var_meta)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem stripping creating variables in {output_filename}", "exc")
        dso.close()
        return 1

    # Add global attributes
    try:
        for a, value in global_meta["global_attributes"].items():
            dso.setncattr(a, value)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem add attribute {a}:{value} to {output_filename}", "exc")
        dso.close()
        return 1

    dso.sync()
    dso.close()

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
