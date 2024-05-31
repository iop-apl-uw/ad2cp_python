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
class SGData(ExtendedDataClass.ExtendedDataClass):
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
    VelENU: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Pitch: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Roll: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))
    Heading: npt.ArrayLike = field(default_factory=(lambda: np.empty(0)))


@dataclass
class ADCPData(ExtendedDataClass.ExtendedDataClass):
    pass


def ADCPReadSGNCF(ds: netCDF4.Dataset, ncf_name: pathlib.Path) -> Tuple[SGData, ADCPData] | Tuple[None, None]:
    sg_data = SGData()
    adcp_data = ADCPData()
    sg_data.VelENU = np.array([ds.variables[f"ad2cp_vel{ii}"][:] for ii in ("X", "Y", "Z")])
    sg_data.VelXYZ = sg_data.VelENU * np.nan
    sg_data.Pitch = dsi.variables["ad2cp_pitch"][:]
    sg_data.Roll = dsi.variables["ad2cp_roll"][:]
    sg_data.Heading = dsi.variables["ad2cp_heading"][:]

    # for k, v in sg_data.items():
    #     if isinstance(v, np.ndarray):
    #         try:
    #             sg_data[k] = ds.variables[k][:]
    #         except Exception:
    #             log_error(f"Failed to load {k}", "exc")
    #             return (None, None)
    #         if k.endswith("_qc"):
    #             sg_data[k] = np.array(list(map(ord, sg_data[k])), np.float64) - ord("0")
    #     elif k == "dive":
    #         sg_data[k] = ds.variables["trajectory"][0]
    #     else:
    #         log_error(f"Don't know how to handle {k}")
    return (sg_data, adcp_data)
