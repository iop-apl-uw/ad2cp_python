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

# import plotly.express as px
import ADCPOpts
from ADCPLog import ADCPLogger, log_critical, log_error, log_warning

DEBUG_PDB = False


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
    ap.add_argument("--debug_pdb", default=False, action="store_true", help="trap on exceptions")
    ap.add_argument(
        "--adcp_log",
        help="Name of output logfile",
        action=ADCPOpts.FullPathAction,
    )
    ap.add_argument("mat_file", help="Matlab file", action=ADCPOpts.FullPathAction)

    ap.add_argument("python_file", help="Python hdf5 file", action=ADCPOpts.FullPathAction)

    args = ap.parse_args()

    global DEBUG_PDB
    DEBUG_PDB = args.debug_pdb

    # Initialize log
    ADCPLogger(args, include_time=True)

    # "/Users/gbs/work/git/adcp_matlab/data_example/sg219_SMODE1/Processed_ADCP/sg219_SMODE1_Aug22_adcp_realtime.mat"
    mat_file = read_matlab(args.mat_file)

    python_file = h5py.File(args.python_file, "r")

    # log_info(np.allclose(python_file["adcp_realtime"]["Z"],  mat_file["adcp_realtime"]["ZZ"]))
    mat_group_name = "adcp"
    # mat_group_name = "adcp_realtime"
    for f_plot, py_grp, py_name, mat_grp, mat_name in (  # noqa: B007
        (False, "adcp_realtime", "Z", mat_group_name, "Z"),
        (False, "adcp_realtime", "Z0", mat_group_name, "Z0"),
        (False, "adcp_realtime", "Svel", mat_group_name, "Svel"),
        (False, "adcp_realtime", "U", mat_group_name, "U"),
        (False, "gps", "XY", "gps", "XY"),
        (False, "glider", "Wmod", "glider", "Wmod"),
        (False, "glider", "UV1", "glider", "UV1"),
        (False, "glider", "ctd_depth", "glider", "ctd_depth"),
        (False, "glider", "ctd_time", "glider", "Mtime"),
        (False, "D", "time", "D", "Mtime"),
        (False, "D", "Z0", "D", "Z0"),
        (False, "D", "UV", "D", "UV"),
        (False, "D", "W", "D", "W"),
        (False, "D", "Z", "D", "Z"),
        (False, "D", "upcast", "D", "upcast"),
        (False, "D", "UVttw_model", "D", "UVttw_model"),
        (False, "D", "Wttw_model", "D", "Wttw_model"),
        (False, "inverse_tmp", "d_adcp", "inverse_tmp", "d_adcp"),
        (False, "inverse_tmp", "z", "inverse_tmp", "z"),
        (False, "inverse_tmp", "TT", "inverse_tmp", "TT"),
        (False, "inverse_tmp", "Z0", "inverse_tmp", "Z0"),
        (False, "inverse_tmp", "upcast", "inverse_tmp", "upcast"),
        (False, "inverse_tmp", "jprof", "inverse_tmp", "jprof"),
        (False, "inverse_tmp", "Av", "inverse_tmp", "Av"),
        (False, "inverse_tmp", "rz", "inverse_tmp", "rz"),
        (False, "inverse_tmp", "iz", "inverse_tmp", "iz"),
        (False, "inverse_tmp", "wz", "inverse_tmp", "wz"),
        (False, "inverse_tmp", "AiM", "inverse_tmp", "AiM"),
        (False, "inverse_tmp", "rrz", "inverse_tmp", "rrz"),
        (False, "inverse_tmp", "iiz", "inverse_tmp", "iiz"),
        (False, "inverse_tmp", "wwz", "inverse_tmp", "wwz"),
        (False, "inverse_tmp", "Ai0", "inverse_tmp", "Ai0"),
        (False, "inverse_tmp", "AiG", "inverse_tmp", "AiG"),
        (False, "inverse_tmp", "G_adcp", "inverse_tmp", "G_adcp"),
        (False, "inverse_tmp", "G_sfc", "inverse_tmp", "G_sfc"),
        (False, "inverse_tmp", "d_sfc", "inverse_tmp", "d_sfc"),
        (False, "inverse_tmp", "G_dac", "inverse_tmp", "G_dac"),
        (False, "inverse_tmp", "d_dac", "inverse_tmp", "d_dac"),
        (False, "inverse_tmp", "dd", "inverse_tmp", "dd"),
        (False, "inverse_tmp", "Do", "inverse_tmp", "Do"),
        (False, "inverse_tmp", "time1", "inverse_tmp", "time1"),
        (False, "inverse_tmp", "time2", "inverse_tmp", "time2"),
        (False, "inverse_tmp", "dd_dnup", "inverse_tmp", "dd_dnup"),
        (False, "inverse_tmp", "Do2", "inverse_tmp", "Do2"),
        (False, "inverse_tmp", "Dv", "inverse_tmp", "Dv"),
        (False, "inverse_tmp", "G_model", "inverse_tmp", "G_model"),
        (False, "inverse_tmp", "d_model", "inverse_tmp", "d_model"),
        (False, "inverse_tmp", "G_deep", "inverse_tmp", "G_deep"),
        (False, "inverse_tmp", "d_deep", "inverse_tmp", "d_deep"),
        (True, "inverse_tmp", "UVttw", "inverse_tmp", "UVttw"),
        (True, "inverse_tmp", "UVocn", "inverse_tmp", "UVocn"),
        (False, "D", "UVocn_solution", "D", "UVocn_solution"),
        (False, "D", "UVttw_solution", "D", "UVttw_solution"),
        (False, "D", "UVveh_solution", "D", "UVveh_solution"),
        (False, "D", "UVocn_adcp", "D", "UVocn_adcp"),
        (False, "D", "UVerr", "D", "UVerr"),
        (False, "D", "Wttw_solution", "D", "Wttw_solution"),
        (False, "D", "Wocn_solution", "D", "Wocn_solution"),
        (False, "profile", "UVocn", "profile", "UVocn"),
        (False, "profile", "Wocn", "profile", "Wocn"),
        (False, "profile", "UVttw_solution", "profile", "UVttw_solution"),
    ):
        # py_var = python_file[py_grp][py_name]
        py_var = np.squeeze(python_file[py_grp][py_name])
        # Note: By applying np.squeeze here, matlab column vectors are converted to row.
        # This has several effects:
        # 1) Masks the differnt orientations between the python and matlab.  A possible downside.
        # 2) Avoids some ammount of warnings for mismatched shapes, that are handled during implict
        #    rebroadcasting inside the allclose and isclose calls
        # 3) Main reason - a row python array that starts with np.nan, followed by values and matlab
        #    column array that starts with np.nan, followed by values will fail np.allclose - something
        #    in the broadcasting maybe, but didn't dig into it further.
        mat_var = np.squeeze(mat_file[mat_grp][mat_name])
        # mat_var = mat_file[mat_grp][mat_name]
        py_shape = np.shape(py_var)
        mat_shape = np.shape(mat_var)
        # Convert any Mtime vars to unix epoch
        if mat_name in ("Mtime", "TT", "time1", "time2"):
            mat_var = (mat_var - 719529) * 86400
        py_str = f"{py_grp}:{py_name}"
        mat_str = f"{mat_grp}:{mat_name}"
        name_str = f"Comparing Py:{py_str:<20} Ma:{mat_str:<20}"
        if py_shape != mat_shape:
            try:
                _ = np.broadcast(py_var, mat_var)
            except ValueError:
                log_error(f"{name_str} shapes don't match ({py_shape}:{mat_shape} and NOT broadcastable")
            else:
                log_warning(f"{name_str} shapes don't match ({py_shape}:{mat_shape}, but is broadcastable")
            # continue
        # Matlab complex is stored as separate real and imaginary
        if py_var.dtype in (np.complex64, np.complex128) and mat_var.dtype != py_var.dtype:
            mat_var = np.squeeze(mat_var)
            tmp = np.zeros(np.shape(mat_var), dtype=py_var.dtype)
            if len(np.shape(mat_var)) == 1:
                for ii in range(np.shape(mat_var)[0]):
                    # np.apply_along_axis(lambda x: x["real"] + 1j * x["imag"], 0, mat_var)
                    tmp[ii] = mat_var[ii]["real"] + 1j * mat_var[ii]["imag"]
                mat_var = tmp
            else:
                if len(np.shape(mat_var)) == 2:
                    for ii in range(np.shape(mat_var)[0]):
                        for jj in range(np.shape(mat_var)[1]):
                            tmp[ii, jj] = mat_var[ii, jj]["real"] + 1j * mat_var[ii, jj]["imag"]
                    mat_var = tmp
                elif len(np.shape(mat_var)) == 0:
                    mat_var = mat_var["real"] + 1j * mat_var["imag"]
                else:
                    log_error("1d case not handled")

        # atol = 1.0
        # while np.allclose(py_var, mat_var, equal_nan=True, atol=atol):
        #     atol = atol / 10
        #     if atol < 1e-10:
        #         break
        # if atol >= 1e-10:
        #     # pdb.set_trace()
        #     close = np.isclose(py_var, mat_var, equal_nan=True, atol=atol)
        #     tot_pts = close.size
        #     not_close_pts = np.count_nonzero(np.logical_not(close))
        #     close_str = f"NOT_CLOSE atol:{atol:g} {not_close_pts}/{tot_pts}"
        #     log_warning(f"{name_str} {close_str}")
        #     # bad_pts = np.nonzero(np.logical_not(close))[0]
        #     bad_pts = np.argwhere(np.logical_not(close))
        #     log_debug("    index:py_var:mat_var")
        #     if len(py_var.shape) == 1:
        #         for ii in bad_pts:
        #             log_debug(f"    [{ii}]:{py_var[ii]}:{mat_var[ii]}")
        #     else:
        #         for ii, jj in bad_pts:
        #             log_debug(f"    [{ii},{jj}]:{py_var[ii,jj]}:{mat_var[ii,jj]}")
        # else:
        #     log_info(f"{name_str} all_close")

        # log_debug(f"{np.shape(py_var)} {np.shape(mat_var)}")

        # From the isclose docs - the test is:
        # absolute(a - b) <= (atol + rtol * absolute(b))

        rtol = 1.0
        atol = 1e-08
        not_close = {}
        # nc_t = collections.namedtuple("nc_t", ("close_b", "close_i"))
        while rtol >= 1e-05:
            not_close[rtol] = np.logical_not(np.isclose(py_var, mat_var, equal_nan=True, rtol=rtol, atol=atol))
            rtol /= 10
        # pdb.set_trace()
        tols = [x for x in not_close]
        if np.count_nonzero(not_close[tols[-1]]):
            # pdb.set_trace()
            # close_str = f"NOT_CLOSE atol:{atol:g} {not_close_pts}/{tot_pts}"
            max_val = np.nanmax(np.abs(py_var))
            mean_val = np.nanmean(np.abs(py_var))
            close_str = f"NOT_CLOSE max:{max_val:<2.3e} mean:{mean_val:<2.3e} tot:{not_close[tols[0]].size:<7d}"
            accum = np.zeros(not_close[tols[-1]].shape, dtype=bool)
            for tol in tols:
                # pdb.set_trace()
                bad_pts_b = np.logical_and(np.logical_not(accum), not_close[tol])
                # bad_pts = np.argwhere(bad_pts_b)
                # log_debug("    index:py_var:mat_var")
                # if len(py_var.shape) == 1:
                #     for ii in bad_pts:
                #         log_debug(f"    [{ii}]:{py_var[ii]}:{mat_var[ii]}")
                # else:
                #     for ii, jj in bad_pts:
                #         log_debug(f"    [{ii},{jj}]:{py_var[ii,jj]}:{mat_var[ii,jj]}")
                nc_count = np.count_nonzero(bad_pts_b)
                accum = np.logical_or(accum, not_close[tol])
                close_str += f" {tol:>1.0e}:{nc_count:<5}"

            # log_warning(f"{name_str} {close_str}")
            print(f"{name_str} {close_str}")
            # Not yet working - need to handle complex
            # if f_plot:
            #    fig = px.scatter(x=py_var)
            #    fig.add_scatter(x=mat_var)
            #    fig.show()

        else:
            # log_info(f"{name_str} all_close rtol:{rtol:g}")
            print(f"{name_str} all_close rtol:{rtol:g}")


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
