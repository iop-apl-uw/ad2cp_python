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

"""Simple basestation extension - starting point for new extensions"""

import os
import pathlib
import pdb
import sys
import time
import traceback
import uuid

import numpy as np

# This needs to be imported before the ADCP files to make sure the Logging infrastructure for the basestation
# is picked up instead of that from the ADCP
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# ruff: noqa: E402
import BaseOpts
import BaseOptsType
import MakeDiveProfiles
import Utils
from BaseLog import BaseLogger, log_debug, log_error, log_info, log_warning

import ADCPConfig
import ADCPUtils

DEBUG_PDB = True


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


def load_additional_arguments():
    """Defines and extends arguments related to this extension.
    Called by BaseOpts when the extension is set to be loaded
    """

    return (
        # Add this module to these options defined in BaseOpts
        ["mission_dir", "netcdf_filename"],
        # Description for any option_group tags used below
        {"adcp": "Nortek ADCP processing for realtime data"},
        # Add these options that are local to this extension
        {
            "adcp_var_meta_file": BaseOptsType.options_t(
                pathlib.Path(__file__).parent.joinpath("config/var_meta.yml"),
                (
                    "Base",
                    "BaseADCP",
                    "BaseADCPMission",
                    "Reprocess",
                    "MakeMissionProfile",
                    "MakeMissionTimeSeries",
                ),
                ("--adcp_var_meta_file",),
                BaseOpts.FullPathlib,
                {
                    "help": "ADCP variable metadata configiuration YAML file",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": BaseOpts.FullPathlibAction,
                },
            ),
            "adcp_global_meta_filename": BaseOptsType.options_t(
                None,
                (
                    "Base",
                    "BaseADCPMission",
                    "Reprocess",
                    "MakeMissionProfile",
                    "MakeMissionTimeSeries",
                ),
                ("--adcp_global_meta_filename",),
                BaseOpts.FullPathlib,
                {
                    "help": "ADCP global metadata configiuration YAML file",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": BaseOpts.FullPathlibAction,
                },
            ),
        },
    )


def main(
    cmdline_args: list[str] = sys.argv[1:],
    instrument_id=None,
    base_opts=None,
    sg_calib_file_name=None,
    dive_nc_file_names=None,
    nc_files_created=None,
    processed_other_files=None,
    known_mailer_tags=None,
    known_ftp_tags=None,
    processed_file_names=None,
    session=None,
):
    """Basestation extension for building whole mission adcp netcdf file

    Returns:
        0 for success (although there may have been individual errors in
            file processing).
        Non-zero for critical problems.

    Raises:
        Any exceptions raised are considered critical errors and not expected
    """
    # pylint: disable=unused-argument
    f_from_cli = False
    if base_opts is None:
        f_from_cli = True
        add_to_arguments, add_option_groups, additional_arguments = load_additional_arguments()

        base_opts = BaseOpts.BaseOptions(
            "Basestation adcp whole mission extensions",
            additional_arguments=additional_arguments,
            add_option_groups=add_option_groups,
            add_to_arguments=add_to_arguments,
            cmdline_args=cmdline_args,
        )

    BaseLogger(base_opts, include_time=True)

    # If called as an extension and no new netcdf filename created, bail out
    if not f_from_cli and not nc_files_created:
        return 0

    if f_from_cli:
        dive_nc_file_names = MakeDiveProfiles.collect_nc_perdive_files(base_opts)

    if not dive_nc_file_names:
        return 0

    global_meta = ADCPConfig.LoadGlobalMeta(base_opts.adcp_global_meta_filename)
    var_meta = ADCPConfig.LoadVarMeta(base_opts.adcp_var_meta_file)

    output_filename = None

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
        "inverse_ocean_velocity_north": False,
        "inverse_ocean_velocity_east": False,
        "inverse_ocean_velocity_vertical": False,
        "inverse_glider_velocity_north": False,
        "inverse_glider_velocity_east": False,
        "inverse_glider_velocity_vertical": False,
        "inverse_time": False,
        "inverse_depth": False,
        "inverse_profile_velocity_north": True,
        "inverse_profile_velocity_east": True,
        "inverse_profile_velocity_vertical": True,
    }

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

    mission_accums = {v: [] for v in mission_vars}

    log_info("Started processing")

    # Collect variable from dive files - load into accumulators
    f_adcp_present = False
    for ncf_name in dive_nc_file_names:
        log_debug(f"Processing {ncf_name}")

        ds = Utils.open_netcdf_file(ncf_name, "r", mask_results=True)

        try:
            vars_tmp = {v: None for v in mission_vars}
            for var_name in vars_tmp:
                vars_tmp[var_name] = ds.variables[var_meta[var_name].nc_varname][:].filled(fill_value=np.nan)
        except KeyError as e:
            msg = (f"Could not load {e} from {ncf_name} - skipping",)
            if f_adcp_present:
                log_info(msg)
            else:
                log_debug(msg)
            continue
        except Exception:
            log_error(f"Problems with loading {ncf_name}", "exc")
            continue

        f_adcp_present = True

        try:
            # This is "inverse_profile_depth" - from the profile_vars
            if depth_grid is None:
                depth_grid = ds.variables["ad2cp_inv_profile_depth"][:].filled(fill_value=np.nan)

            glider_dive = ds.variables["trajectory"][:].filled(fill_value=np.nan)[0]
            glider_longitude = ds.variables["longitude"][:].filled(fill_value=np.nan)
            glider_latitude = ds.variables["latitude"][:].filled(fill_value=np.nan)
            glider_temperature = ds.variables["temperature"][:].filled(fill_value=np.nan)
            glider_salinity = ds.variables["salinity"][:].filled(fill_value=np.nan)
            glider_ctd_depth = ds.variables["ctd_depth"][:].filled(fill_value=np.nan)
        except KeyError as e:
            log_warning(f"Could not load {e} from {ncf_name} - skipping")
            continue
        except Exception:
            log_error(f"Problems with loading {ncf_name}", "exc")
            continue

        if not output_filename:
            output_filename = pathlib.Path(base_opts.mission_dir).joinpath(
                f"{ADCPUtils.GetMissionStr(ds).replace(' ', '_').replace('-', '_')}_adcp_profile_timeseries.nc"
            )

        if not copy_ncattrs:
            for attr in global_nc_attrs:
                copy_ncattrs[attr] = ds.getncattr(attr)

        for var_name in vars_tmp:
            mission_accums[var_name].append(vars_tmp[var_name])

        mission_dive.append(glider_dive)
        mission_dive.append(glider_dive)

        lon_min = min(lon_min, np.nanmin(glider_longitude))
        lon_max = max(lon_max, np.nanmax(glider_longitude))
        lat_min = min(lat_min, np.nanmin(glider_latitude))
        lat_max = max(lat_max, np.nanmax(glider_latitude))

        imax = np.argmax(glider_ctd_depth)
        mission_lon.append(np.nanmean(glider_longitude[: imax + 1]))
        mission_lon.append(np.nanmean(glider_longitude[imax:]))
        mission_lat.append(np.nanmean(glider_latitude[: imax + 1]))
        mission_lat.append(np.nanmean(glider_latitude[imax:]))
        mission_temperature.append(
            ADCPUtils.bindata(glider_ctd_depth[: imax + 1], glider_temperature[: imax + 1], depth_grid)[0]
        )
        mission_temperature.append(ADCPUtils.bindata(glider_ctd_depth[imax:], glider_temperature[imax:], depth_grid)[0])
        mission_salinity.append(
            ADCPUtils.bindata(glider_ctd_depth[: imax + 1], glider_salinity[: imax + 1], depth_grid)[0]
        )
        mission_salinity.append(ADCPUtils.bindata(glider_ctd_depth[imax:], glider_salinity[imax:], depth_grid)[0])

        # TODO - need a similar reduction in depth for the profile variables

    if not f_adcp_present:
        log_info("No ADCP found - finished processing")
        return 0

    mission_var_arrs = {}
    non_nans = []
    for k, is_profile in mission_vars.items():
        try:
            mission_var_arrs[k] = np.hstack(mission_accums[k])
            if is_profile:
                tmp = np.logical_not(np.isnan(mission_var_arrs[k]))
                # log_info(f"{k}:{np.shape(tmp)}:{tmp.dtype}")
                non_nans.append(tmp)
        except ValueError:
            log_warning(f"No data found for variable {k} - skipping")
            continue

    # Find the deepest data for profile
    deepest_i = max(np.nonzero(np.logical_or.reduce(non_nans))[0]) + 1

    # Associate variables with meta data and trim
    # data to the deepest observation
    ad2cp_variable_mapping = {
        "inverse_profile_depth": depth_grid[:deepest_i],
        "inverse_profile_dive": np.hstack(mission_dive),
        "inverse_profile_longitude": np.hstack(mission_lon),
        "inverse_profile_latitude": np.hstack(mission_lat),
    }
    for v, is_profile in mission_vars.items():
        if is_profile:
            ad2cp_variable_mapping[v] = mission_var_arrs[v][:deepest_i, :]
        else:
            ad2cp_variable_mapping[v] = mission_var_arrs[v]

    ad2cp_variable_mapping["profile_temperature"] = np.vstack(mission_temperature).T[:deepest_i, :]
    ad2cp_variable_mapping["profile_salinity"] = np.vstack(mission_salinity).T[:deepest_i, :]

    dso = ADCPUtils.open_netcdf_file(output_filename, "w")
    if dso is None:
        log_error(f"Could not open {output_filename}")
        return 1
    log_info(f"Writing out {output_filename}")

    try:
        ADCPUtils.CreateNCVars(dso, ad2cp_variable_mapping, var_meta)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem stripping creating variables in {output_filename}", "exc")
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

    log_info("Finished processing")
    return 0


if __name__ == "__main__":
    retval = 0
    try:
        retval = main()
    except SystemExit:
        pass
    except Exception:
        if DEBUG_PDB:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        sys.stderr.write(f"Exception in main ({traceback.format_exc()})\n")

    sys.exit(retval)
