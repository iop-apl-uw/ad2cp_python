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
ADCPUtils.py - Utility functions
"""

import pathlib
import sys

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy

from ADCPLog import log_error


def check_versions() -> int:
    """
    Checks for minimum versions of critical libraries and python.

    Returns:
        0 for no issue, 1 for version outside minimum
    """

    # Python version
    if sys.version_info < (3, 10):
        log_error("Python version 3.10 at minimum is required")
        return 1

    return 0


def open_netcdf_file(ncf_name: pathlib.Path) -> None | netCDF4.Dataset:
    try:
        ds = netCDF4.Dataset(ncf_name, "r")
    except Exception:
        log_error(f"Failed to open {ncf_name}", "exc")
        return None
    ds.set_auto_mask(False)
    return ds


def intnan(y: npt.NDArray[np.float64]) -> None:
    """Interpolate over NaNs.  Leading and trailing NaNs are replaced wtih the first/last
    non-nan value. Works on one dimensional items only
    """
    sz = y.size
    good_mask = np.logical_not(np.isnan(y))
    i_good = np.nonzero(good_mask)
    # Fill in the tails
    y[0 : np.min(i_good)] = y[np.min(i_good)]
    y[np.max(i_good) + 1 :] = y[np.max(i_good)]
    # Regen mask now that the tails are filled in
    good_mask = np.logical_not(np.isnan(y))
    i_good = np.nonzero(good_mask)
    t = np.arange(sz)
    f = scipy.interpolate.interp1d(
        t[good_mask],
        y[good_mask],
        # bounds_error=False,
        # fill_value="extrapolate",
    )
    y[np.logical_not(good_mask)] = f(t[np.logical_not(good_mask)])
