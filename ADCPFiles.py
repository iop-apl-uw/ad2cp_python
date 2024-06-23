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
from typing import Any, Tuple

import netCDF4
import numpy as np
import numpy.typing as npt

import ADCPConfig
import ADCPUtils
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


class SaveToHDF5:
    def save_to_hdf5(self, group_name: str, hdf) -> None:
        """Persist the dataclass fields to a group in a HDF5 file"""
        grp = hdf.create_group(group_name)
        for var_n in self.__dataclass_fields__:
            grp.create_dataset(var_n, data=getattr(self, var_n))


@dataclass
class ADCPRealtimeData(ExtendedDataClass.ExtendedDataClass, SaveToHDF5):
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
    Svel: float | npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
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
    def adcp_namemapping(self):
        return {
            "ad2cp_velX": ("U", lambda x: fetch_var(x).T),
            "ad2cp_velY": ("V", lambda x: fetch_var(x).T),
            "ad2cp_velZ": ("W", lambda x: fetch_var(x).T),
            "ad2cp_pitch": ("pitch", fetch_var),
            "ad2cp_roll": ("roll", fetch_var),
            "ad2cp_heading": ("heading", fetch_var),
            "ad2cp_soundspeed": ("Svel", lambda x: fetch_var(x) / 10.0),
            "ad2cp_cellSize": ("cellSize", fetch_var),
            "ad2cp_blanking": ("blanking", fetch_var),
            "ad2cp_pressure": ("pressure", fetch_var),
            "ad2cp_time": ("time", fetch_var),
            # "ad2cp_foobar": ("foobar", fetch_var),  # for testing
        }

    def init(self, ds: netCDF4.Dataset, ncf_name: pathlib.Path) -> None:
        # Load variables from dataset
        for var_n in self.adcp_namemapping():
            if var_n not in ds.variables:
                raise KeyError(f"Could not find {var_n} in {ncf_name}")
            self[self.adcp_namemapping()[var_n][0]] = self.adcp_namemapping()[var_n][1](ds.variables[var_n])

        # Tilt factor : the Z component of (0,0,1) vector transformed
        # adcp_realtime.TiltFactor =  cos(pi*adcp_realtime.pitch/180).*cos(pi*adcp_realtime.roll/180);
        self.TiltFactor = np.cos(np.pi * self.pitch / 180.0) * np.cos(np.pi * self.roll / 180)
        # adcp_realtime.Range = (1:size(adcp_realtime.U,1))'*adcp_realtime.cellSize/1000+adcp_realtime.blanking/100;
        self.Range = np.arange(1, np.shape(self.U)[0] + 1) * self.cellSize / 1000.0 + self.blanking / 100.0


@dataclass
class GPSData(ExtendedDataClass.ExtendedDataClass, SaveToHDF5):
    """Glider data from the seaglider netcdf file"""

    log_gps_time: npt.NDArray[np.complex128] = field(default_factory=(lambda: np.empty(0)))
    #
    # Derived/calculated
    #

    # GPS lat/lon on a complex plane
    LL: npt.NDArray[np.complex128] = field(default_factory=(lambda: np.empty(0)))
    # Distance from origin point on complex plane
    XY: npt.NDArray[np.complex128] = field(default_factory=(lambda: np.empty(0)))


@dataclass
class SGData(ExtendedDataClass.ExtendedDataClass, SaveToHDF5):
    """Glider data from the seaglider netcdf file and derived values"""

    # From the netcdf file
    dive: int = 0
    longitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    latitude: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    longitude_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    latitude_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    ctd_depth: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    ctd_time: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    vert_speed: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    vert_speed_gsm: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    sound_velocity: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    log_gps_time: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    log_gps_lat: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    log_gps_lon: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    #
    # Derived
    #

    #  horizontal velocity of glider, with DAC in it, in ENU coordinate
    ##UV0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    #   vertical velocity of glider, from dp/dp
    ##W0: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # horizontal velocity of glider, without DAC in it, in ENU coordinate
    ##UV1: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # total horizontal displacement of the glider, with DAC, from the active model.
    ##xy: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    # vertical speed of the glider from active model
    Wmod: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))
    # Time for last GPS fix before dive
    time0: np.float64 = 0
    # Time for first GPS fix after dive
    time1: np.float64 = 0

    def load_vars(self) -> list[str]:
        """List of variables to be loaded from the netcdf file"""
        return [
            "dive",
            "longitude",
            "latitude",
            "longitude_gsm",
            "latitude_gsm",
            "ctd_depth",
            "ctd_time",
            "vert_speed",
            "vert_speed_gsm",
            "sound_velocity",
            "log_gps_time",
            "log_gps_lat",
            "log_gps_lon",
        ]

    def init(self, ds: netCDF4.Dataset, ncf_name: pathlib.Path, param: ADCPConfig.Params, gps: GPSData) -> None:
        for var_n in self.load_vars():
            v = self[var_n]
            if isinstance(v, np.ndarray):
                try:
                    self[var_n] = ds.variables[var_n][:]
                except Exception:
                    log_error(f"Failed to load {var_n}", "exc")
                    continue
                if var_n.endswith("_qc"):
                    self[var_n] = np.array(list(map(ord, self[var_n])), np.float64) - ord("0")
            elif var_n == "dive":
                self[var_n] = ds.variables["trajectory"][0]
            else:
                log_error(f"Don't know how to handle {var_n}")
        for var_n in self.load_vars():
            if var_n.endswith("_qc"):
                continue
            qc_var = f"{var_n}_qc"
            if qc_var in self:
                # TODO - apply QC to data
                pass

        # Derived/calculated

        # gps.LL = glider.log_gps_lon+1i*glider.log_gps_lat;
        # gps.Mtime = glider.log_gps_time/86400+datenum(1970,1,1);
        gps.LL = self.log_gps_lon + 1j * self.log_gps_lat
        gps.log_gps_time = self.log_gps_time

        # % get one (two?) more GPS fix from the next dive (if available)
        # fn_g1 = sprintf('p%d%04d.nc',param.sg,dive+1);
        # if exist([param.dir_profiles, fn_g1],'file')
        #   G1 = read_netcdf([param.dir_profiles, fn_g1],{'log_gps_lon','log_gps_lat','log_gps_time'});
        #   if ~isempty(G1)
        #     ige = 2; % extra GPS fixe(s)
        #     gps.LL = [gps.LL;G1.log_gps_lon(ige)+1i*G1.log_gps_lat(ige)];
        #     gps.Mtime = [gps.Mtime;G1.log_gps_time(ige)/86400+datenum(1970,1,1)];
        #   end
        # end
        next_dive_ncf = ncf_name.parent / f"p{param.sg:03d}{self.dive+1:04d}.nc"
        if next_dive_ncf.exists():
            ds = ADCPUtils.open_netcdf_file(next_dive_ncf)
            if ds:
                gps.LL = np.array((gps.LL, ds["log_gps_lon"][:2] + 1j * ds["log_gps_lon"][:2]))
                gps.log_gps_time = np.array((gps.log_gps_time, ds["log_gps_time"][:2]))

        # LLgsm = glider.longitude_gsm+1i*glider.latitude_gsm;
        # LL = glider.longitude+1i*glider.latitude; % best estimate? includes dac.
        LLgsm = self.longitude_gsm + 1j * self.latitude_gsm
        LL = self.longitude + 1j * self.latitude

        # glider.Mtime = glider.ctd_time/86400+datenum(1970,1,1);

        # % last GPS before the dive...
        # ig1 = find(gps.Mtime<glider.Mtime(1),1,'last');
        # % first GPS after the dive...
        # ig2 = find(gps.Mtime>glider.Mtime(end),1,'first');
        # % for_each_field gps '*=*([ig1 ig2]);'
        # LL0 = gps.LL(ig1); % origin...
        # glider.Mtime0 = gps.Mtime(ig1);
        # glider.Mtime1 = gps.Mtime(ig2);
        # gps.XY = latlon2xy(gps.LL,LL0)*1e3; % m

        # Last GPS before the dive
        ig1 = np.where(gps.log_gps_time < self.ctd_time[0])[0].max()
        # First GPS after the dive
        ig2 = np.where(gps.log_gps_time > self.ctd_time[-1])[0].min()
        # Origin
        LL0 = gps.LL[ig1]
        self.time0 = gps.log_gps_time[ig1]
        self.time1 = gps.log_gps_time[ig2]
        gps.XY = ADCPUtils.latlon2xy(gps.LL, LL0) * 1e3  # in meters

        # if strcmp(param.VEHICLE_MODEL,'gsm')
        #   xy = latlon2xy(LLgsm,LL0);
        #   glider.Wmod = glider.vert_speed_gsm/100;
        #   % from the flight model:
        #   h = course_interp(glider.time/86400+datenum(1970,1,1),glider.eng_head,glider.Mtime)+glider.magnetic_variation;
        #   glider.UV1 = sqrt(glider.speed_gsm.^2-glider.vert_speed_gsm.^2)/100.*cos((90-h)*pi/180) + 1i*sqrt(glider.speed_gsm.^2-glider.vert_speed_gsm.^2)/100.*sin((90-h)*pi/180);
        # elseif strcmp(param.VEHICLE_MODEL, 'FlightModel')
        #   xy = latlon2xy(LL,LL0);
        #   glider.Wmod = glider.vert_speed/100;
        #   % from the flight model:
        #   h = course_interp(glider.time/86400+datenum(1970,1,1),glider.eng_head,glider.Mtime)+glider.magnetic_variation;
        #   glider.UV1 = sqrt(glider.speed.^2-glider.vert_speed.^2)/100.*cos((90-h)*pi/180) + 1i*sqrt(glider.speed.^2-glider.vert_speed.^2)/100.*sin((90-h)*pi/180);
        # end

        h = ADCPUtils.course_interp(self.time, self.eng_head, self.ctd_time) + self.magnetic_variation
        if param.VEHICLE_MODEL == "gsm":
            xy = ADCPUtils.latlon2xy(LLgsm, LL0)
            self.Wmod = self.vert_speed_gsm / 100.0
            vert_speed_var = "vert_speed_gsm"
            speed_var = "speed_gsm"
        else:
            xy = ADCPUtils.latlon2xy(LL, LL0)
            self.Wmod = self.vert_speed / 100.0
            vert_speed_var = "vert_speed"
            speed_var = "speed"

        # from the flight model:
        self.UV1 = np.sqrt(self[speed_var] ** 2 - self[vert_speed_var] ** 2) / 100 * np.cos(
            (90 - h) * np.pi / 180
        ) + 1j * np.sqrt(self[speed_var] ** 2 - self[vert_speed_var] ** 2) / 100 * sin((90 - h) * np.pi / 180)

        ###
        # NYI and untested from here down
        ###

        # glider.xy = black_filt(xy,5);
        # clear xy
        self.xy = ADCPUtils.black_filt(xy, 5)

        # UV0 = (diff(glider.xy)*1e3)./diff(glider.Mtime)/86400; % estimated horizontal glider speed from flight model (including dac).
        # glider.UV0 = [UV0(1) ; 0.5*(UV0(1:end-1)+UV0(2:end)) ; UV0(end)];
        # clear UV0

        # % Vertical speed.
        # W0 = -(diff(glider.ctd_depth))./diff(glider.Mtime)/86400; % estimated vertical glider speed
        # W0 = [W0(1) ; 0.5*(W0(1:end-1)+W0(2:end)) ; W0(end)];
        # % smooth W0
        # glider.W0 = conv2P(W0,ones(3,1)/3);
        # clear W0

        # % % time gap -
        # glider.UV0(abs(glider.UV0)>1)=NaN;
        # ii = find(diff(glider.Mtime)*86400>60);
        # glider.UV0([ii ii+1])=NaN; % larger than a minute


# For full ADCP data sets
@dataclass
class ADCPData(ExtendedDataClass.ExtendedDataClass, SaveToHDF5):
    pass


def ADCPReadSGNCF(
    ds: netCDF4.Dataset, ncf_name: pathlib.Path, param: ADCPConfig.Params
) -> Tuple[SGData, GPSData, ADCPRealtimeData]:
    """ """
    adcp_realtime_data = ADCPRealtimeData()
    adcp_realtime_data.init(ds, ncf_name)

    glider = SGData()
    gps = GPSData()
    glider.init(ds, ncf_name, param, gps)

    # adcp_raw = ADCPData()
    # adcp_raw.init()

    # TODO in 2019, missing fields - this might be needed out here
    # if ~isfield(adcp_realtime,'cellSize')
    #  adcp_realtime.Range=double(adcp_raw.Range);
    # else
    #  adcp_realtime.Range = (1:size(adcp_realtime.U,1))'*adcp_realtime.cellSize/1000+adcp_realtime.blanking/100;
    # end

    return (glider, gps, adcp_realtime_data)
