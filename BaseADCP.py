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

import time

import BaseOpts
import Utils
from BaseLog import BaseLogger, log_info

DEBUG_PDB = False


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
        base_opts = BaseOpts.BaseOptions(
            "Basestation extension for processing ADCP data files",
        )

    BaseLogger(base_opts)  # initializes BaseLog

    for dive_nc_file_name in dive_nc_file_names:
        log_info(f"Processing {dive_nc_file_name}")

        ncf = Utils.open_netcdf_file(dive_nc_file_name, "a")

        # Simple test for ability to update a netcdf file
        curr_time_str = time.strftime("%H:%M:%S %d %b %Y %Z", time.gmtime(time.time()))
        ncf.__setattr__("BaseADCP", curr_time_str)
        ncf.variables["sg_cal_pitchbias"].assignValue(time.time())

        ncf.sync()
        ncf.close()

    return 0
