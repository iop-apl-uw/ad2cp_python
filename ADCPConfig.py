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

import numpy as np
import numpy.typing as npt
import yaml

import ExtendedDataClass
from ADCPLog import log_error


@dataclass
class OptionsProcessing(ExtendedDataClass.ExtendedDataClass):
    # Gilder number int
    sg: int = 0
    # Size of vertical grid bins used in inverse solution
    dz: float = 5.0
    # surface blank (if W_SURFACE ~=0, glider velocity is zero AND ADCP data is ignored above that depth
    sfc_blank: float = 3.0
    # restrict range bins
    gz: npt.ArrayLike = field(default_factory=(lambda: np.zeros(0)))
    # param.index_bins = 1:15
    index_bins: npt.ArrayLike = field(default_factory=(lambda: np.arange(15)))
    # 'gsm' or 'FlightModel'
    VEHICLE_MODEL: str = "FlightModel"
    # Use the glider pressure sensor
    use_glider_pressure = True


def ProcessConfigFile(config_file_name: pathlib.PosixPath) -> OptionsProcessing:
    """Return the default set of options, with updated by a config file if present

    Args:
        config_file_name: Fully qualified path to config file or empty

    Returns:
        OptionsProcessing object
    """

    opt_p = OptionsProcessing()
    if not config_file_name:
        return opt_p
    try:
        with open(config_file_name, "r") as fi:
            cfg_dict = yaml.safe_load(fi.read())
    except Exception:
        log_error(f"Failed to process {config_file_name}", "exc")
        return opt_p

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
