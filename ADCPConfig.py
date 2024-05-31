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
ADCPConfig.py - Routines to process SG ADCP config file
"""

import pathlib
from dataclasses import dataclass, field

# import numpy as np
# import numpy.typing as npt
import yaml

import ExtendedDataClass
from ADCPLog import log_error


@dataclass
class OptionsProcessing(ExtendedDataClass.ExtendedDataClass):
    # Gilder number as a string
    sg: str = "000"
    # List of dives to process
    dives: list[int] = field(default_factory=list)
    # Directory for the profiles
    dir_profiles: pathlib.Path = pathlib.Path(".").expanduser().absolute()
    dir_rawfiles: pathlib.Path = pathlib.Path(".").expanduser().absolute()
    mission: str = ""
    # channels:list[str]= ['temp','shear']
    channels: list[str] = field(default_factory=lambda: ["temp", "shear"])
    # realtime_raw:list[int] = [1, 1]
    # [1 0]: realtime only, [0 1] = raw only = up, [1 1] = both realtime and raw (if present)
    realtime_raw: list[int] = field(default_factory=lambda: [1, 1])
    file_label: str = ""

    directions: int = 1  # 1 = dn, 2 = up, 1:2 = both dn and up.

    plot_figures: bool = False  # which ensemble in this dive (or climb) should we plot?

    verbose: int = 2  # 0 is nothing, 1 is everything, 2 is limited.
    speed_source: str = "hdm"  # Do we use the glide slope model ('gsm'), or the full hydro model  ('hdm', default)?

    # Slope (log-log) of the noise for the temperature gradient, or shear spectrum
    slope_noise_tmicro: float = 1.5
    slope_noise_smicro: float = 1
    s_noise_floor: float = 50
    # Shear signal ad-hoc scaling.
    s_gain_empirical: float = 4
    # Processing options for the raw data, if they are available
    #    Note: This is needed for processing regular in time, or to overwirte
    #        what was done in real time...
    navg: int = 4
    nfft: int = 256
    despike_flag: int = 1  # despike
    despikethreshold: int = 2
    # In the MLE fits, how many loops do I do to get the best range of
    # wavenumbers do I do (3).
    n_loops_k_range: int = 3
    # When we have the raw data, we have the option to process the data on a
    # prescribed vertical grid, where ensembles are created that include all
    # the in that depth grid. This will have a different Navg for each ensemble.

    # UNIFORM IN TIME: set to NaN. The processing will be regular in time,
    rawdata_z_grid: float = float("nan")
    rawdata_dz_grid: float = float("nan")
    # UNIFORM IN DEPTH: If we want depth bins: speficy the grid here.
    #   rawdata_z_grid  = [1:1:5 6:2:200]'    # center depth of the estimates
    #   rawdata_dz_grid = ones(size(rawdata_z_grid))*2
    #   rawdata_z_grid(1:5)=1
    max_shear_sigvar: float = 1e9
    max_temp_sigvar: float = 1e9


def ProcessConfigFile(config_file_name: pathlib.PosixPath) -> OptionsProcessing | None:
    try:
        with open(config_file_name, "r") as fi:
            cfg_dict = yaml.safe_load(fi.read())
    except Exception:
        log_error(f"Failed to process {config_file_name}", "exc")
        return None

    opt_p = OptionsProcessing()
    for k, v in cfg_dict.items():
        try:
            if k not in opt_p:
                log_error(f"Unknown config variable {k} - skipping")
                continue
            if isinstance(v, type(opt_p[k])):
                opt_p[k] = v
            elif isinstance(opt_p[k], pathlib.Path):
                opt_p[k] = pathlib.Path(v).expanduser().absolute()
            elif isinstance(opt_p[k], float):
                opt_p[k] = float(v)
            elif isinstance(opt_p[k], int):
                opt_p[k] = int(v)
            else:
                log_error(f"Do not know how to handle {k} {v} from {config_file_name}")
            # TODO - lots of mis-config can fall through the above processing look at tightening
            # this up with more refinement
        except ValueError:
            log_error(f"Failed to convert {k}:{v} in {config_file_name} to type {type(opt_p[k])}")
        except Exception:
            log_error(f"Failed to handle {k} in {config_file_name}", "exc")
    return opt_p
