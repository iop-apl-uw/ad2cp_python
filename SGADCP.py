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

import netCDF4
import numpy as np

import ADCP
import ADCPConfig
import ADCPFiles
import ADCPOpts
import ADCPRealtime
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_debug, log_error, log_info

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
    adcp_opts = ADCPOpts.ADCPOptions("Command line driver Seaglider ADCPx data processing")

    # Initialize log
    ADCPLogger(adcp_opts, include_time=True)

    if ADCPUtils.check_versions():
        return 1

    # Read the config file
    param = ADCPConfig.ProcessConfigFile(adcp_opts.config_file)
    if not param:
        return 1

    # param.gz = (0:param.dz:1000)';
    # TODO - not sure the transposition is the right thing, but its in the matlab code
    param.gz = np.arange(0, 1000 + param.dz, param.dz)[:, np.newaxis]

    # Mission-long corrections - not used in the 2024 version of the code
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

    # timeseries_adcp.time = [];
    # timeseries_adcp.Ux = [];
    # timeseries_adcp.Uy = [];
    # timeseries_adcp.Uz = [];

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
        try:
            ds = netCDF4.Dataset(ncf_name, "r")
        except Exception:
            log_error(f"Failed to open {ncf_name}", "exc")
            continue
        ds.set_auto_mask(False)

        if not param.sg:
            param.sg = ds.glider

        # Read real-time and adcp_raw (if present)
        try:
            glider, gps, adcp_realtime_data = ADCPFiles.ADCPReadSGNCF(ds, ncf_name)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem loading data", "exc")
            continue

        ds.close()
        # Transform velocites to instrument frame
        try:
            ADCPRealtime.TransformToInstrument(adcp_realtime_data)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem transforming compass data", "exc")
            continue

        # TODO: Add a "cone plot" here - x y z is lon, lat, depth and u v w is from ADCP
        # TODO: Compare frame Ux, Uy, Uz with matlab output

        # Clean up adcp data
        try:
            ADCP.CleanADCP(adcp_realtime_data, glider, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem cleaning realtime adcp data", "exc")
            continue

        # TODO: Compare ??? with matlab output

        # TODO - weights need to be loadable from the config
        # Setup weights
        weights = None

        # Perfrom inverse
        try:
            D, profile, variables_for_plot = ADCP.Inverse5(adcp_realtime_data, gps, glider, weights, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem performing inverse calculation", "exc")
            continue

        # Plot output

        # Save back generated columns

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
