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

import collections
import os
import pathlib
import pdb
import sys
import time
import traceback
import uuid

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

DEBUG_PDB = False


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


def main(cmdline_args: list[str] = sys.argv[1:]) -> int:
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
        "Command line driver Seaglider ADCPx data processing",
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

    global_nc_attrs = [
        "platform_id",
        "source",
        "platform",
        "summary",
        "project",
        "glider",
        "mission",
        "base_station_version",
        "base_station_micro_version",
    ]
    copy_ncattrs = {}

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

    if adcp_opts.include_glider_vars:
        glider_vars = {
            "ctd_time": var_t(lambda x: x.time, "glider_ctd_time"),
            "ctd_depth": var_t(lambda x: x.time, "glider_ctd_depth"),
            "temperature": var_t(lambda x: x.time, "glider_temperature"),
            "salinity": var_t(lambda x: x.time, "glider_salinity"),
            "longitude": var_t(lambda x: x.time, "glider_longitude"),
            "latitude": var_t(lambda x: x.time, "glider_latitude"),
        }
    else:
        glider_vars = {}

    # Extents
    lat_max = -90.0
    lat_min = 90.0
    lon_max = -180.0
    lon_min = 180.0

    # Whole mission accumulators
    depth_grid = None
    mission_dive = []
    mission_lon = []
    mission_lat = []
    mission_temperature = []
    mission_salinity = []

    mission_accums = {}
    for k in mission_vars:
        mission_accums[k] = []

    glider_accums = {}
    for k in glider_vars:
        glider_accums[k] = []

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

    if not dive_nc_filenames:
        log_error(f"No profile netCDF files found in {adcp_opts.mission_dir} - bailing out")
        return 1

    for ncf_name in dive_nc_filenames:
        log_info(f"Processing {ncf_name}")
        ds = ADCPUtils.open_netcdf_file(ncf_name)
        if ds is None:
            continue

        if not param.sg_id:
            param.sg_id = ds.glider

        if not copy_ncattrs:
            for attr in global_nc_attrs:
                copy_ncattrs[attr] = ds.getncattr(attr)

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
            if ADCP.CleanADCP(adcp_realtime, glider, param) is None:
                log_error("Failed cleaning realtime adcp data")
                continue
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

        if depth_grid is None:
            depth_grid = profile.z

        mission_dive.append(glider.dive)
        mission_dive.append(glider.dive)

        lon_min = min(lon_min, np.nanmin(glider.longitude))
        lon_max = max(lon_max, np.nanmax(glider.longitude))
        lat_min = min(lat_min, np.nanmin(glider.latitude))
        lat_max = max(lat_max, np.nanmax(glider.latitude))

        imax = np.argmax(glider.ctd_depth)
        mission_lon.append(np.nanmean(glider.longitude[: imax + 1]))
        mission_lon.append(np.nanmean(glider.longitude[imax:]))
        mission_lat.append(np.nanmean(glider.latitude[: imax + 1]))
        mission_lat.append(np.nanmean(glider.latitude[imax:]))
        mission_temperature.append(
            ADCPUtils.bindata(glider.ctd_depth[: imax + 1], glider.temperature[: imax + 1], depth_grid)[0]
        )
        mission_temperature.append(ADCPUtils.bindata(glider.ctd_depth[imax:], glider.temperature[imax:], depth_grid)[0])
        mission_salinity.append(
            ADCPUtils.bindata(glider.ctd_depth[: imax + 1], glider.salinity[: imax + 1], depth_grid)[0]
        )
        mission_salinity.append(ADCPUtils.bindata(glider.ctd_depth[imax:], glider.salinity[imax:], depth_grid)[0])

        for k, v in mission_vars.items():
            mission_accums[k].append(v.accessor(profile))

        for k, v in glider_vars.items():
            glider_accums[k].append(v.accessor(glider))

        # Debug save out - for comparision with other processing
        # TODO - need a better name - match with input file
        if adcp_opts.save_details:
            output_details_filename = ncf_name.with_suffix(".hdf5")
            with h5py.File(output_details_filename, "w") as hdf:
                log_info(f"Writing out {output_details_filename}")
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

    glider_var_arrs = {}
    for k in glider_vars:
        glider_var_arrs[k] = np.hstack(glider_accums[k])

    # Find the deepest data
    deepest_i = max(np.nonzero(np.logical_or.reduce(non_nans))[0]) + 1

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

    ad2cp_variable_mapping["profile_temperature"] = np.vstack(mission_temperature).T[:deepest_i, :]
    ad2cp_variable_mapping["profile_salinity"] = np.vstack(mission_salinity).T[:deepest_i, :]

    for k, v in glider_vars.items():
        ad2cp_variable_mapping[v.var_meta_name] = glider_var_arrs[k]

    dso = ADCPUtils.open_netcdf_file(output_filename, "w")
    if dso is None:
        log_error(f"Could not open {output_filename}")
        return 1
    log_info(f"Writing out {output_filename}")

    try:
        ADCPUtils.CreateNCVars(dso, ad2cp_variable_mapping, var_meta)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem creating variables in {output_filename}", "exc")
        dso.close()
        return 1

    try:
        ADCPUtils.WriteParamsWeights(dso, weights, param, var_meta)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem creating config variables in {output_filename}", "exc")
        dso.close()
        return 1

    # Add global attributes
    dso.setncattr("geospatial_lon_min", lon_min)
    dso.setncattr("geospatial_lon_max", lon_max)
    dso.setncattr("geospatial_lat_min", lat_min)
    dso.setncattr("geospatial_lat_max", lat_max)

    dso.setncattr("geospatial_vertical_min", np.nanmin(depth_grid[:deepest_i]))
    dso.setncattr("geospatial_vertical_max", np.nanmax(depth_grid[:deepest_i]))

    now_date = time.strftime("%Y-%m-%dT%H:%m:%SZ", time.gmtime(time.time()))
    dso.setncattr("date_created", now_date)
    dso.setncattr("date_modified", now_date)
    dso.setncattr("file_version", "1.0")
    dso.setncattr("uuid", str(uuid.uuid1()))

    try:
        for attr, value in copy_ncattrs.items():
            dso.setncattr(attr, value)

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
    ret_val = 1
    try:
        ret_val = main()
    except Exception:
        DEBUG_PDB_F()
        log_critical("Unhandled exception in main -- exiting", "exc")

    sys.exit(ret_val)
