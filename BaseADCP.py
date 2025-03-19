# -*- python-fmt -*-

## Copyright (c) 2024, 2025  University of Washington.
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

"""Basestation extension for microstructure post-processing"""

import argparse
import os
import pathlib
import pdb
import shutil
import sys
import traceback

import numpy as np
from pyproj import Geod

# This needs to be imported before the ADCP files to make sure the Logging infrastructure for the basestation
# is picked up instead of that from the ADCP
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# ruff: noqa: E402
import BaseNetCDF
import BaseOpts
import BaseOptsType
import MakeDiveProfiles
import Utils
from BaseLog import BaseLogger, log_error, log_info

import ADCP
import ADCPConfig
import ADCPFiles
import ADCPRealtime
import ADCPUtils

DEBUG_PDB = False


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


def GenerateNewLatLons(ds, D):
    gps_lons = ds.variables["log_gps_lon"][:]
    gps_lats = ds.variables["log_gps_lat"][:]
    gps_times = ds.variables["log_gps_time"][:]

    # TODO - Need to get the lat/lon for the positions before GPS2
    # Do that by doing a separate inegration starting at GPS1 for the surface drifts and
    # combining the two vectors

    # i_dive = np.logical_and(gps_times[1] <= D.time, D.time <= gps_times[2])
    i_dive = np.logical_and(D.time >= gps_times[1], D.time <= gps_times[2])
    total_speed = D.UVocn_solution[i_dive] + D.UVttw_solution[i_dive]
    duration = np.hstack((D.time[i_dive][0] - gps_times[1], np.diff(D.time[i_dive])))

    # pdb.set_trace()
    azimuths = np.degrees(np.pi / 2.0 - np.angle(total_speed))
    distances = np.abs(total_speed) * duration

    geod = Geod(ellps="WGS84")
    lons = [gps_lons[1]]
    lats = [gps_lats[1]]
    times = [gps_times[1]]
    # Needs to be one point at a time in a loop
    for ii in range(len(azimuths)):
        lon, lat, _ = geod.fwd(lons[-1], lats[-1], azimuths[ii], distances[ii])
        lons.append(lon)
        lats.append(lat)
        times.append(D.time[i_dive][ii])
    lons.append(gps_lons[2])
    lats.append(gps_lats[2])
    times.append(gps_times[2])

    # Perform quick back check - from the test data,
    # log_info("Check for lat/lon calculation")
    # for ii in range(len(lons) - 1):
    #    _, _, dist = geod.inv(lons[ii], lats[ii], lons[ii + 1], lats[ii + 1])
    #    log_info(f"{ii}:{dist}")
    az, _, dist = geod.inv(lons[0], lats[0], lons[-2], lats[-2])
    last_az, _, last_dist = geod.inv(lons[0], lats[0], lons[-1], lats[-1])
    log_info(f"{az:.2f}:{dist:.2f} {last_az:.2f}:{last_dist:.2f}")

    # Using 2d local approximation
    # m_per_deg = 111120.0
    # lon_fac = np.cos(np.radians((gps_lats[1] + gps_lats[2]) / 2))

    # alt_lons = [gps_lons[1]]
    # alt_lats = [gps_lats[1]]
    # alt_times = [gps_times[1]]
    # for ii in range(len(total_speed)):
    #     lon, lat, _ = geod.fwd(alt_lons[-1], alt_lats[-1], azimuths[ii], distances[ii])
    #     alt_lons.append(alt_lons[-1] + total_speed[ii].real * duration[ii] / m_per_deg * lon_fac)
    #     alt_lats.append(alt_lats[-1] + total_speed[ii].imag * duration[ii] / m_per_deg)
    #     alt_times.append(D.time[i_dive][ii])
    # alt_lons.append(gps_lons[2])
    # alt_lats.append(gps_lats[2])
    # alt_times.append(gps_times[2])

    # for ii in range(len(lons) - 1):
    #     _, _, dist = geod.inv(alt_lons[ii], alt_lats[ii], alt_lons[ii + 1], alt_lats[ii + 1])
    #     log_info(f"{ii}:{dist}")

    return (np.array(lons), np.array(lats), np.array(times))


def convert_to_base_meta(var_meta_d, nc_info, f_include):
    """Converts a adcp var_meta entry to a basestation var_meta dict"""

    attribs_d = {}

    for k, v in iter(var_meta_d.nc_attribs):
        if k == "FillValue":
            attribs_d["_FillValue"] = v
        else:
            attribs_d[k] = v

    # MMT and MMP require the time vector to be marked as "included"
    return (f_include, var_meta_d.nc_type, attribs_d, nc_info)


def init_extension(module_name: str, base_opts: BaseOpts.BaseOptions | None = None, init_dict=None) -> int:
    """
    init_sensor

    Returns:
        -1 - error in processing
         0 - success
    """

    # The only purpose of this function is to add metadata into the basestation's system to
    # allow MakeMissionTimeSeries and MakeMisionProfile to incorporte and act on the
    # ADCP timeseries solutions (along to plotting of those solutions).

    # Plotting for the ADCP profile solutions are handled by a dedicated
    # Basestation plotting routine.  This is because MMT and MMP work strictly on timeseries
    # data.

    # List of possible variables and which are included in MMT and MMP by default
    ad2cp_vars = {
        # Timeseries data - adcp solutions at the gliders position
        "inverse_time": True,  # Note: this has to be True, as it is required my MMP to do binning
        "inverse_depth": True,
        "inverse_ocean_velocity_north": True,
        "inverse_ocean_velocity_east": True,
        "inverse_glider_velocity_north": False,
        "inverse_glider_velocity_east": False,
        "inverse_ocean_velocity_vertical": False,
        "inverse_glider_velocity_vertical": False,
    }

    if init_dict is None:
        log_error("No datafile supplied for init_extension - version mismatch?")
        return -1

    # TODO - remove when BaseOpts FullPath is converted to pathlib
    base_opts.adcp_var_meta_file = pathlib.Path(base_opts.adcp_var_meta_file)
    var_meta = ADCPConfig.LoadVarMeta(base_opts.adcp_var_meta_file)

    # CONSIDER - this should data driven from the above list
    # hardcode it for now
    BaseNetCDF.register_sensor_dim_info(
        "ad2cp_inv_glider_point_info",
        "ad2cp_inv_glider_point",
        "ad2cp_inv_glider_time",  # A time variable must be registered for MMP to run - see note on "inverse_time" above
        True,
        # If an instrumnent varaible is need, change to and see note below "ad2cp_inv_glider",
        None,
    )

    adds_dict = {
        #
        # Something like this could be added if an instrument variable were needed
        #
        # "ad2cp_inv_glider": [
        #     False,
        #     "c",
        #     {
        #         # "long_name": "underway thermosalinograph",
        #         # "nodc_name": "thermosalinograph",
        #         # "make_model": "unpumped RBR Legato",
        #     },
        #     BaseNetCDF.nc_scalar,
        # ],  # always scalar
    }
    for var_name, f_include in ad2cp_vars.items():
        new_md = convert_to_base_meta(
            var_meta[var_name],
            ("ad2cp_inv_glider_point_info",),
            f_include,
        )
        adds_dict[var_meta[var_name].nc_varname] = new_md

    init_dict[module_name] = {
        "netcdf_metadata_adds": adds_dict,
    }


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
            "adcp_config_file": BaseOptsType.options_t(
                "",
                (
                    "Base",
                    "BaseADCP",
                    "Reprocess",
                    "MakeMissionProfile",
                    "MakeMissionTimeSeries",
                ),
                ("--adcp_config_file",),
                BaseOpts.FullPath,
                {
                    "help": "ADCP configiuration YAML file",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": BaseOpts.FullPathAction,
                },
            ),
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
                BaseOpts.FullPath,
                {
                    "help": "ADCP variable metadata configiuration YAML file",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": BaseOpts.FullPathAction,
                },
            ),
            "adcp_include_frame_vel": BaseOptsType.options_t(
                False,
                (
                    "Base",
                    "BaseADCP",
                    "Reprocess",
                    "MakeMissionProfile",
                    "MakeMissionTimeSeries",
                ),
                ("--adcp_include_frame_vel",),
                bool,
                {
                    "help": "Include the realtime data adcp frame co-ordinates",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": argparse.BooleanOptionalAction,
                },
            ),
            "adcp_include_updated_latlon": BaseOptsType.options_t(
                False,
                (
                    "Base",
                    "BaseADCP",
                    "Reprocess",
                    "MakeMissionProfile",
                    "MakeMissionTimeSeries",
                ),
                ("--adcp_include_updated_latlon",),
                bool,
                {
                    "help": "Include estimated lat/lons based on adcp estimates",
                    "section": "adcp",
                    "option_group": "adcp",
                    "action": argparse.BooleanOptionalAction,
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
):
    """Basestation extension for running microstructure processing

    Returns:
        0 for success (although there may have been individual errors in
            file processing).
        Non-zero for critical problems.

    Raises:
        Any exceptions raised are considered critical errors and not expected

    """
    # pylint: disable=unused-argument
    if base_opts is None:
        add_to_arguments, add_option_groups, additional_arguments = load_additional_arguments()
        base_opts = BaseOpts.BaseOptions(
            "Basestation extension for processing ADCP data files",
            additional_arguments=additional_arguments,
            add_option_groups=add_option_groups,
            add_to_arguments=add_to_arguments,
            cmdline_args=cmdline_args,
        )

    global DEBUG_PDB
    DEBUG_PDB = base_opts.debug_pdb

    BaseLogger(base_opts)  # initializes BaseLog

    if ADCPUtils.check_versions():
        return 1

    if not base_opts.mission_dir and hasattr(base_opts, "netcdf_filename") and base_opts.netcdf_filename:
        # Called from CLI with a single argument
        dive_nc_file_names = [base_opts.netcdf_filename]
    elif base_opts.mission_dir:
        if nc_files_created is not None:
            # Called from MakeDiveProfiles as extension
            dive_nc_file_names = nc_files_created
        elif not dive_nc_file_names:
            # Called from CLI to process whole mission directory
            # Collect up the possible files
            dive_nc_file_names = MakeDiveProfiles.collect_nc_perdive_files(base_opts)
    else:
        log_error("Either mission_dir or netcdf_file must be specified")
        return 1

    # Read the config file
    param, weights = ADCPConfig.ProcessConfigFile(base_opts.adcp_config_file)
    if not param:
        return 1

    # TODO - remove when BaseOpts FullPath is converted to pathlib
    base_opts.adcp_var_meta_file = pathlib.Path(base_opts.adcp_var_meta_file)
    var_meta = ADCPConfig.LoadVarMeta(base_opts.adcp_var_meta_file)

    for dive_nc_file_name in dive_nc_file_names:
        dive_nc_file_name = pathlib.Path(dive_nc_file_name)

        try:
            ds = Utils.open_netcdf_file(dive_nc_file_name)
        except Exception:
            log_error(f"Failed to open {dive_nc_file_name} - skipping update", "exc")
            continue

        if "ad2cp_time" not in ds.variables:
            ds.close()
            continue

        log_info(f"Processing {dive_nc_file_name}")

        if not param.sg:
            param.sg = ds.glider

        # Read real-time
        try:
            glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, dive_nc_file_name, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem loading data", "exc")
            ds.close()
            continue

        param.time_limits = np.array((np.min(gps.log_gps_time), np.max(gps.log_gps_time)))

        # Transform velocites to instrument frame
        try:
            ADCPRealtime.TransformToInstrument(adcp_realtime)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem transforming compass data", "exc")
            ds.close()
            continue

        # Clean up adcp data
        try:
            ADCP.CleanADCP(adcp_realtime, glider, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem cleaning realtime adcp data", "exc")
            ds.close()
            continue

        # Perfrom inverse
        try:
            D, profile, variables_for_plot, inverse_tmp = ADCP.Inverse(adcp_realtime, gps, glider, weights, param, {})
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem performing inverse calculation", "exc")
            ds.close()
            continue

        # U == EW == real
        # V == NS == imag

        # This table drives the output in the netcdf file.  Keys here, are the keys in the
        # var_meta dict loaded in from yml
        ad2cp_variable_mapping = {
            # Glider time variables
            # Time for each solution
            "inverse_time": D.time,
            # Depth of glider
            "inverse_depth": D.Z0,
            # Inverse solution of horizontal ocean velocity at the location of the glider.
            "inverse_ocean_velocity_north": D.UVocn_solution.imag,
            "inverse_ocean_velocity_east": D.UVocn_solution.real,
            # Inverse solution of the total glider velocity (throught the water + ocean drift)
            # "inverse_glider_total_velocity_north": D.UVveh_solution.imag,
            # "inverse_glider_total_velocity_east": D.UVveh_solution.real,
            # Inverse solution of horizontal glider velocity through the water
            "inverse_glider_velocity_north": D.UVttw_solution.imag,
            "inverse_glider_velocity_east": D.UVttw_solution.real,
            # Inverse solution of vertical glider velocity at the location of the glider
            "inverse_ocean_velocity_vertical": D.Wocn_solution,
            # Inverse solution of vertical ocean velocity at the location of the glider
            "inverse_glider_velocity_vertical": D.Wttw_solution,
            # ADCP profile variables
            "inverse_profile_depth": profile.z,
            "inverse_profile_velocity_north": profile.UVocn.imag,
            "inverse_profile_velocity_east": profile.UVocn.real,
            "inverse_profile_velocity_vertical": profile.Wocn,
        }

        if base_opts.adcp_include_frame_vel:
            # ADCP velocities in glider frame coordinates
            ad2cp_variable_mapping |= {
                "ad2cp_frame_Ux": adcp_realtime.Ux,
                "ad2cp_frame_Uy": adcp_realtime.Uy,
                "ad2cp_frame_Uz": adcp_realtime.Uz,
                "ad2cp_frame_time": adcp_realtime.time,
            }

        # Generate new lat/lon values
        if base_opts.adcp_include_updated_latlon:
            new_lons = new_lats = new_times = None
            try:
                new_lons, new_lats, new_times = GenerateNewLatLons(ds, D)
            except Exception:
                DEBUG_PDB_F()
                log_error(f"Problem computing updated lat/lons from {dive_nc_file_name}", "exc")
            else:
                # Returns include the starting and ending GPS position, so drop
                # those when writing out
                ad2cp_variable_mapping |= {
                    "ad2cp_longitude": new_lons[1:-1],
                    "ad2cp_latitude": new_lats[1:-1],
                    "ad2cp_lonlat_time": new_times[1:-1],
                }

        strip_meta = {}
        for key_n in ad2cp_variable_mapping:
            strip_meta[key_n] = var_meta[key_n]

        # Create temporary output netcdf file
        tmp_filename = dive_nc_file_name.with_suffix(".tmpnc")
        try:
            dso = Utils.open_netcdf_file(tmp_filename, "w")
        except Exception:
            log_error(f"Failed to open tempfile {tmp_filename} - skipping update to {dive_nc_file_name}", "exc")
            ds.close()
            continue

        try:
            ADCPUtils.StripVars(ds, dso, strip_meta)
        except Exception:
            DEBUG_PDB_F()
            log_error(f"Problem stripping old variables from {dive_nc_file_name}", "exc")
            ds.close()
            continue

        try:
            ADCPUtils.CreateNCVars(dso, ad2cp_variable_mapping, var_meta)
        except Exception:
            DEBUG_PDB_F()
            log_error(f"Problem stripping creating variables in {tmp_filename}", "exc")
            ds.close()
            dso.close()
            continue

        ds.close()
        dso.sync()
        dso.close()
        shutil.move(tmp_filename, dive_nc_file_name)

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
        sys.stderr.write(f"Exception in main {traceback.format_exc()}\n")

    sys.exit(retval)
