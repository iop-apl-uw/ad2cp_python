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
from dataclasses import dataclass, field
from typing import Any, Tuple

import pdb

import netCDF4
import numpy as np
import numpy.typing as npt

import ExtendedDataClass
from ADCPLog import log_error


def fetch_var(x: netCDF4._netCDF4.Variable) -> Any:
    """Helper function for data fetch"""
    # For netCDF4 datasets, singletons have an empty tuple for the shape
    if not x.shape:
        # Singleton
        return x[0]
    else:
        # Array
        return x[:]
    return x


@dataclass
class ADCPRealtimeData(ExtendedDataClass.ExtendedDataClass):
    """ADCP Realtime data from the seaglider netcdf file"""

    #
    # From realtime data
    #

    # Velocities in earth co-ordinates (changed during cleanup)
    U: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    V: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    W: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Velocities in earth co-ordinates - as reported by instrument
    U0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    V0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    W0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Compass output from ADCP
    pitch: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    roll: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    heading: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    pressure: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    time: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    # ADCP's sound velocity, later glider's soundvelocty interpolated on the adcp grid
    SVel: float | npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    blanking: float = 0
    cellSize: float = 0

    #
    # Derived/calculated
    #

    # Velocities in instrument frame co-ordinates
    Ux: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    Uy: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    Uz: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # the Z component of (0,0,1) vector transformed
    TiltFactor: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Ranges of each cell
    Range: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Depth - either from ADCP or from glider - switch dependent
    Z0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Depth for each cell for each ensemble
    Z: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    # Add needed vars are in this table
    adcp_namemapping = {
        "ad2cp_velX": ("U", fetch_var),
        "ad2cp_velY": ("V", fetch_var),
        "ad2cp_velZ": ("W", fetch_var),
        "ad2cp_pitch": ("pitch", fetch_var),
        "ad2cp_roll": ("roll", fetch_var),
        "ad2cp_heading": ("heading", fetch_var),
        "ad2cp_soundspeed": ("SVel", lambda x: fetch_var(x) / 10.0),
        "ad2cp_cellSize": ("cellSize", fetch_var),
        "ad2cp_blanking": ("blanking", fetch_var),
        "ad2cp_pressure": ("pressure", fetch_var),
        "ad2cp_time": ("time", fetch_var),
        # "ad2cp_foobar": ("foobar", fetch_var),  # for testing
    }

    def init(self, ds: netCDF4.Dataset, ncf_name: pathlib.Path) -> None:
        # Load variables from dataset
        for var_n in self.adcp_namemapping:
            if var_n not in ds.variables:
                raise KeyError(f"Could not find {var_n} in {ncf_name}")
            self[self.adcp_namemapping[var_n][0]] = self.adcp_namemapping[var_n][1](ds.variables[var_n])

        # Tilt factor : the Z component of (0,0,1) vector transformed
        # adcp_realtime.TiltFactor =  cos(pi*adcp_realtime.pitch/180).*cos(pi*adcp_realtime.roll/180);
        self.TiltFactor = np.cos(np.pi * self.pitch / 180.0) * np.cos(np.pi * self.roll / 180)
        # adcp_realtime.Range = (1:size(adcp_realtime.U,1))'*adcp_realtime.cellSize/1000+adcp_realtime.blanking/100;
        self.Range = np.arange(1, np.shape(self.U)[1] + 1) * self.cellSize / 1000.0 + self.blanking / 100.0


@dataclass
class SGData(ExtendedDataClass.ExtendedDataClass):
    """Glider data from the seaglider netcdf file and derived values"""

    load_vars = [
        "longitude",
        "latitude",
        "longitude",
        "latitude",
        "ctd_depth",
        "ctd_time",
        "vert_speed",
        "sound_velocity",
    ]

    # From the netcdf file
    longitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    latitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    longitude_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    latitude_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    ctd_depth: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    ctd_time: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    vert_speed: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    sound_velocity: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    #
    # Derived
    #

    #  horizontal velocity of glider, with DAC in it, in ENU coordinate
    UV0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    #   vertical velocity of glider, from dp/dp
    W0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # horizontal velocity of glider, without DAC in it, in ENU coordinate
    UV1: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # total horizontal displacement of the glider, with DAC.
    xy: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    def init(self, ds: netCDF4.Dataset, ncf_name: pathlib.Path) -> None:
        for var_n in self.load_vars:
            v = self[var_n]
            if isinstance(v, np.ndarray):
                try:
                    self[var_n] = ds.variables[var_n][:]
                except Exception:
                    log_error(f"Failed to load {var_n}", "exc")
                    return (None, None)
                if var_n.endswith("_qc"):
                    self[var_n] = np.array(list(map(ord, self[var_n])), np.float64) - ord("0")
            elif var_n == "dive":
                self[var_n] = ds.variables["trajectory"][0]
            else:
                log_error(f"Don't know how to handle {var_n}")
        for var_n in self.load_vars:
            if var_n.endswith("_qc"):
                continue
            qc_var = f"{var_n}_qc"
            if qc_var in self:
                # TODO - apply QC to data
                pass

    # temperature: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # temperature_qc: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # ctd_depth: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # ctd_time: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # salinity: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # salinity_qc: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # latitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # longitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # speed: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # speed_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # speed_qc: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # depth_avg_curr_east: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # depth_avg_curr_north: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # dive: float = 0


@dataclass
class GPSData(ExtendedDataClass.ExtendedDataClass):
    """Glider data from the seaglider netcdf file"""


# For full ADCP data sets
@dataclass
class ADCPData(ExtendedDataClass.ExtendedDataClass):
    pass


def ADCPReadSGNCF(ds: netCDF4.Dataset, ncf_name: pathlib.Path) -> Tuple[SGData, GPSData, ADCPRealtimeData]:
    """ """
    adcp_realtime_data = ADCPRealtimeData()
    adcp_realtime_data.init(ds, ncf_name)

    glider = SGData()
    glider.init(ds, ncf_name)

    gps = GPSData()
    # gps.init(ds, ncf_name)

    # adcp_raw = ADCPData()
    # adcp_raw.init()

    # in 2019, missing fields - this might be needed out here
    # if ~isfield(adcp_realtime,'cellSize')
    #  adcp_realtime.Range=double(adcp_raw.Range);
    # else
    #  adcp_realtime.Range = (1:size(adcp_realtime.U,1))'*adcp_realtime.cellSize/1000+adcp_realtime.blanking/100;
    # end

    return (glider, gps, adcp_realtime_data)
