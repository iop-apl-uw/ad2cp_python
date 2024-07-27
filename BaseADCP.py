# -*- python-fmt -*-

## Copyright (c) 2024  University of Washington.
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

import os
import pathlib
import pdb
import sys
import traceback

import numpy as np

import ADCP
import ADCPConfig
import ADCPFiles

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# ruff: noqa: E402
import BaseOpts
import BaseOptsType
import MakeDiveProfiles
import Utils
from BaseLog import BaseLogger, log_error, log_info

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
            "adcp_config_file": BaseOptsType.options_t(
                "",
                (
                    "Base",
                    "BaseADCP",
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
        },
    )


def main(
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
        )

    BaseLogger(base_opts)  # initializes BaseLog

    if not base_opts.mission_dir and hasattr(base_opts, "netcdf_filename") and base_opts.netcdf_filename:
        dive_nc_file_names = [base_opts.netcdf_filename]
    elif base_opts.mission_dir:
        if nc_files_created is not None:
            # Called from MakeDiveProfiles
            dive_nc_file_names = nc_files_created
        elif not dive_nc_file_names:
            # Collect up the possible files
            dive_nc_file_names = MakeDiveProfiles.collect_nc_perdive_files(base_opts)
    else:
        log_error("Either mission_dir or netcdf_file must be specified")
        return 1

    # Read the config file
    param, weights = ADCPConfig.ProcessConfigFile(base_opts.adcp_config_file)
    if not param:
        return 1

    for dive_nc_file_name in dive_nc_file_names:
        dive_nc_file_name = pathlib.Path(dive_nc_file_name)

        log_info(f"Processing {dive_nc_file_name}")

        ds = Utils.open_netcdf_file(dive_nc_file_name)
        if ds is None:
            continue

        if not param.sg:
            param.sg = ds.glider

        # Read real-time
        try:
            glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, dive_nc_file_name, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem loading data", "exc")
            continue

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

        # Perfrom inverse
        try:
            D, profile, variables_for_plot, inverse_tmp = ADCP.Inverse(adcp_realtime, gps, glider, weights, param, {})
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem performing inverse calculation", "exc")
            continue

        # From basestation extension, push the following into the netcdf:

        # D.time
        # D.z0
        # D.UVocn_solution
        # D.UVttw_solution
        # D.Wttw_solution
        # D.Wocn_solution
        # profile.z
        # profile.UVocn
        # profile.Wocn
        #
        # Plus Ux, Uy, and Uz (for FMS and plotting)

        # Run processing using existing netcdf file

        # Duplicate file into tmp name, eliminating adcp vectors and replacing with new

        # Over write exisitng

        # ncf = Utils.open_netcdf_file(dive_nc_file_name, "a")

        # Simple test for ability to update a netcdf file
        # curr_time_str = time.strftime("%H:%M:%S %d %b %Y %Z", time.gmtime(time.time()))
        # ncf.__setattr__("BaseADCP", curr_time_str)
        # ncf.variables["sg_cal_pitchbias"].assignValue(time.time())

        # ncf.sync()
        # ncf.close()

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
        sys.stderr.write("Exception in main (%s)\n" % traceback.format_exc())

    sys.exit(retval)
