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
CompareToMat.py - Compares single dive output between matlab and python implimentations
"""

import argparse
import os
import pdb
import sys
import time
import traceback

import h5py
import numpy as np

import ADCPOpts
from ADCPLog import ADCPLogger, log_critical, log_debug, log_error, log_info, log_warning

DEBUG_PDB = True


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


# From https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python
# Loads matlab 7.3v files - handles the conversion from FORTRAN to C order
def read_matlab(filename):
    def conv(path=""):
        p = path or "/"
        paths[p] = ret = {}
        for k, v in f[p].items():
            if type(v).__name__ == "Group":
                ret[k] = conv(f"{path}/{k}")  # Nested struct
                continue
            v = v[()]  # It's a Numpy array now
            if v.dtype == "object":
                # HDF5ObjectReferences are converted into a list of actual pointers
                # The extra check for np.bytes_ excludes vars that are bytearrays (strings)
                ret[k] = [r.dtype.type is not np.bytes_ and r and paths.get(f[r].name, f[r].name) for r in v.flat]
            else:
                # Matrices and other numeric arrays
                ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
        return ret

    paths = {}
    with h5py.File(filename, "r") as f:
        return conv()


def main() -> None:
    app_name, _ = os.path.splitext(__file__)

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")
    ap.add_argument("--debug", default=False, action="store_true", help="enable debug output")
    ap.add_argument(
        "--adcp_log",
        help="Name of output logfile",
        action=ADCPOpts.FullPathAction,
    )
    ap.add_argument("mat_file", help="Matlab file", action=ADCPOpts.FullPathAction)

    ap.add_argument("python_file", help="Python hdf5 file", action=ADCPOpts.FullPathAction)

    args = ap.parse_args()

    # Initialize log
    ADCPLogger(args, include_time=True)

    # "/Users/gbs/work/git/adcp_matlab/data_example/sg219_SMODE1/Processed_ADCP/sg219_SMODE1_Aug22_adcp_realtime.mat"
    mat_file = read_matlab(args.mat_file)

    python_file = h5py.File(args.python_file, "r")

    # log_info(np.allclose(python_file["adcp_realtime"]["Z"],  mat_file["adcp_realtime"]["ZZ"]))
    mat_group_name = "adcp"
    for py_grp, py_name, mat_grp, mat_name in (
        ("adcp_realtime", "Z", mat_group_name, "Z"),
        ("adcp_realtime", "Z0", mat_group_name, "Z0"),
        ("adcp_realtime", "Svel", mat_group_name, "Svel"),
        ("adcp_realtime", "U", mat_group_name, "U"),
    ):
        py_var = python_file[py_grp][py_name]
        mat_var = mat_file[mat_grp][mat_name]
        py_shape = np.shape(py_var)
        mat_shape = np.shape(mat_var)
        name_str = f"Comparing Python:{py_grp}:{py_name}, Matlab:{mat_grp}:{mat_name}"
        if py_shape != mat_shape:
            try:
                _ = np.broadcast(py_var, mat_var)
            except ValueError:
                log_error(f"{name_str} shapes don't match ({py_shape}:{mat_shape} and NOT broadcastable")
            else:
                log_warning(f"{name_str} shapes don't match ({py_shape}:{mat_shape}, but is broadcastable")
            # continue
        atol = 1.0
        while np.allclose(python_file[py_grp][py_name], mat_file[mat_grp][mat_name], equal_nan=True, atol=atol):
            atol = atol / 10
            if atol < 1e-10:
                break
        if atol >= 1e-10:
            close = np.isclose(python_file[py_grp][py_name], mat_file[mat_grp][mat_name], equal_nan=True, atol=atol)
            tot_pts = close.size
            not_close_pts = np.count_nonzero(np.logical_not(close))
            close_str = f"NOT_CLOSE atol:{atol:g} {not_close_pts}/{tot_pts}"
            log_warning(f"{name_str} {close_str}")
        else:
            log_info(f"{name_str} all_close")

        log_debug(f"{np.shape(python_file[py_grp][py_name])} {np.shape(python_file[py_grp][py_name])}")


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
