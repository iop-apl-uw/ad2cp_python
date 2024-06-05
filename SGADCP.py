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

import ADCPConfig
import ADCPFiles
import ADCPOpts
import ADCPRealtime
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_debug, log_error, log_info

DEBUG_PDB = True


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
    if not adcp_opts.config_file.exists():
        log_error(f"Config file {adcp_opts.config_file} does not exist")
        return 1

    opts_p = ADCPConfig.ProcessConfigFile(adcp_opts.config_file)
    if opts_p is None:
        return 1

    log_debug(opts_p)

    dive_nc_filenames: list[pathlib.Path] = []
    for dd in opts_p.dives:
        try:
            tmp_name = opts_p.dir_profiles.joinpath(f"p{opts_p.sg}{dd:04d}.nc")
            if tmp_name.exists():
                dive_nc_filenames.append(tmp_name)
            else:
                log_info(f"{tmp_name} not found - skipping")
        except Exception:
            log_error(f"Failed processing dive {dd}", "exc")

    for ncf_name in dive_nc_filenames:
        log_info(f"Processing {ncf_name}")
        try:
            ds = netCDF4.Dataset(ncf_name, "r")
        except Exception:
            log_error(f"Failed to open {ncf_name}", "exc")
            continue
        ds.set_auto_mask(False)

        # Read real-time and adcp_raw (if present)
        try:
            adcp_realtime_data, adcp_data = ADCPFiles.ADCPReadSGNCF(ds, ncf_name)
        except Exception:
            log_error("Failed to read {ncf_name}", "exc")
            continue

        ds.close()

        # Transform velocites to instrument frame
        try:
            ADCPRealtime.TransformToInstrument(adcp_realtime_data)
        except Exception:
            log_error("Failed to converting to glider frame {ncf_name}", "exc")
            continue

        # Add a "cone plot" here - x y z is lon, lat, depth and u v w is from ADCP
        log_debug(adcp_realtime_data.U, adcp_realtime_data.V, adcp_realtime_data.W)

    return 0


if __name__ == "__main__":
    # Force to be in UTC
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        ret_val = main()
    except Exception:
        if DEBUG_PDB:
            _, __, traceb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(traceb)
        log_critical("Unhandled exception in main -- exiting", "exc")

    sys.exit(ret_val)
