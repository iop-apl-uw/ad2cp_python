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
ADCPFiles.py - Routines to process Seaglider ADCP files
"""

import pathlib
import pdb
from dataclasses import dataclass, field
from typing import Tuple

import netCDF4
import numpy as np
import numpy.typing as npt

import ExtendedDataClass
from ADCPLog import log_error


@dataclass
class ADCPRealtimeData(ExtendedDataClass.ExtendedDataClass):
    """ADCP Realtime data from the seaglider netcdf file"""

    # Velocities in earth co-ordinates
    U: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    V: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    W: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # Velocities in instrument frame co-ordinates
    Ux: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Uy: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Uz: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # Compass output from ADCP
    Pitch: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Roll: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Heading: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))


@dataclass
class SGData(ExtendedDataClass.ExtendedDataClass):
    """Glider data from the seaglider netcdf file"""

    pass
    # temperature: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # temperature_qc: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # ctd_depth: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # ctd_time: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # salinity: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # salinity_qc: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # latitude: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # longitude: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # speed: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # speed_gsm: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # speed_qc: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # depth_avg_curr_east: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # depth_avg_curr_north: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    # dive: float = 0


@dataclass
class SGData(ExtendedDataClass.ExtendedDataClass):
    """Glider data from the seaglider netcdf file"""


# @dataclass
# class ADCPData(ExtendedDataClass.ExtendedDataClass):
#    pass


def ADCPReadSGNCF(
    ds: netCDF4.Dataset, ncf_name: pathlib.Path
) -> Tuple[SGDAta, GPSData, ADCPRealtimeData] | Tuple[None, None, None]:
    adcp_realtime_data = ADCPRealtimeData()
    glider = SGData()
    gps = GPSData()
    try:
        adcp_realtime_data.U = ds.variables["ad2cp_velX"][:]
        adcp_realtime_data.V = ds.variables["ad2cp_velY"][:]
        adcp_realtime_data.W = ds.variables["ad2cp_velZ"][:]
        adcp_realtime_data.Pitch = ds.variables["ad2cp_pitch"][:]
        adcp_realtime_data.Roll = ds.variables["ad2cp_roll"][:]
        adcp_realtime_data.Heading = ds.variables["ad2cp_heading"][:]
    except Exception:
        log_error("Failed loading realtime data", "exc")
        return (None, None)

    return (adcp_realtime_data, adcp_data)
