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
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import yaml

import ExtendedDataClass

if "BaseLog" in sys.modules:
    from BaseLog import log_error
else:
    from ADCPLog import log_error


def init_helper(
    config_obj: Any, config_file_name: pathlib.PosixPath, cfg_dict: Dict, section_name: str, convert_fn: Any
) -> None:
    if section_name in cfg_dict and isinstance(cfg_dict[section_name], dict):
        for k, v in cfg_dict[section_name].items():
            try:
                if k not in convert_fn():
                    log_error(f"Unknown config variable {k} - skipping")
                    continue
                config_obj[k] = convert_fn()[k](v)
            except ValueError:
                log_error(f"Failed to convert {k}:{v} in {config_file_name} to type {type(config_obj[k])}")
            except Exception:
                log_error(f"Failed to handle {k} in {config_file_name}", "exc")


@dataclass
class Params(ExtendedDataClass.ExtendedDataClass):
    # Gilder number int
    sg: int = field(default=0)
    # Size of vertical grid bins used in inverse solution
    dz: float = field(default=5.0)
    # surface blank (if W_SURFACE ~=0, glider velocity is zero AND ADCP data is ignored above that depth
    sfc_blank: float = field(default=3.0)
    # param.index_bins = 1:15
    index_bins: npt.ArrayLike = field(default_factory=(lambda: np.arange(15)))
    # 'gsm' or 'FlightModel'
    VEHICLE_MODEL: str = field(default="FlightModel")
    # Use the glider pressure sensor
    use_glider_pressure: bool = field(default=True)

    # Not configurable - set from glider data
    # CONSIDER - move to glider object?

    # Min and max time for each data set
    time_limits: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    # restrict range bins
    gz: npt.ArrayLike = field(default_factory=(lambda: np.zeros(0)))

    def params_conversion(self) -> Dict:
        return {
            "sg": int,
            "dz": float,
            "sfc_blank": float,
            "index_bins": lambda x: np.array(x, np.int32),
            "VEHICLE_MODEL": str,
            "use_glider_pressure": bool,
        }

    def init(self, cfg_dict: dict, config_file_name: pathlib.PosixPath) -> None:
        init_helper(self, config_file_name, cfg_dict, "params", self.params_conversion)

        # Transposition - was in the matlab code, but doesn't look right for the python case
        # self.gz = np.arange(0, 1000 + self.dz, self.dz)[:, np.newaxis]
        self.gz = np.arange(0, 1000 + self.dz, self.dz)


@dataclass
class Weights(ExtendedDataClass.ExtendedDataClass):
    W_MEAS: float = field(default=1)  # measurement weight: ~ 1/(0.05);
    OCN_SMOOTH: float = field(default=1)  # 100/param.dz; % smoothness factors
    VEH_SMOOTH: float = field(default=1)  # smoothness factors
    W_DAC: float = field(default=4)  # Weight for the total barotropic constraint (gps)
    W_MODEL: float = field(default=1)  # vehicle flight model weight (when we don't have ADCP data).
    W_MODEL_DAC: float = field(default=2)  # vehicle model dac weight
    W_SURFACE: float = field(default=1)  # Weight for surface constraint (gps - surface drift)
    W_OCN_DNUP: float = field(default=2)  # Weight for down and up ocean profile to be the same.
    W_deep: float = field(default=0.1)  # Weight to make velocities below W_deep_z0 small
    W_deep_z0: float = field(default=500)
    W_MODEL_bottom: bool = field(default=False)  # Bool to set ttw speed zero at the bottom

    def params_conversion(self) -> dict:
        return {
            "W_MEAS": float,
            "OCN_SMOOTH": float,
            "VEH_SMOOTH": float,
            "W_DAC": float,
            "W_MODEL": float,
            "W_MODEL_DAC": float,
            "W_SURFACE": float,
            "W_OCN_DNUP": float,
            "W_deep": float,
            "W_deep_z0": float,
        }

    def init(self, cfg_dict: dict, config_file_name: pathlib.PosixPath) -> None:
        init_helper(self, config_file_name, cfg_dict, "weights", self.params_conversion)


def ProcessConfigFile(config_file_name: pathlib.PosixPath) -> Tuple[Params, Weights]:
    """Return the default set of options, with updated by a config file if present

    Args:
        config_file_name: Fully qualified path to config file or empty

    Returns:
        Tuple of Params object and Weights object
    """

    cfg_dict = {}
    if config_file_name:
        try:
            with open(config_file_name, "r") as fi:
                cfg_dict = yaml.safe_load(fi.read())
        except Exception:
            log_error(f"Failed to process {config_file_name} - continuing", "exc")

    opt_p = Params()
    opt_p.init(cfg_dict, config_file_name)

    weights = Weights()
    weights.init(cfg_dict, config_file_name)

    return (opt_p, weights)


class AttributeDict(dict):
    """Allow dot access for dictionaries"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def LoadVarMeta(var_meta_file):
    var_meta = {}
    try:
        with open(var_meta_file, "r") as fi:
            var_meta_tmp = yaml.safe_load(fi)

        for k in var_meta_tmp:
            var_meta[k] = AttributeDict(var_meta_tmp[k])
    except Exception:
        log_error(f"Could not process {var_meta_file}", "exc")
    return var_meta


def LoadGlobalMeta(global_meta_file_local):
    global_meta = {}

    global_meta_file = pathlib.Path(__file__).parent.joinpath("config/global_meta.yml")

    try:
        with open(global_meta_file, "r") as fi:
            global_meta = yaml.safe_load(fi)
    except Exception:
        log_error(f"Could not process {global_meta_file}", "exc")

    if global_meta_file_local is not None:
        try:
            with open(global_meta_file_local, "r") as fi:
                global_meta_local = yaml.safe_load(fi)
        except Exception:
            log_error(f"Could not process {global_meta_file_local}", "exc")
        global_meta = MergeDict(global_meta, global_meta_local)

    return global_meta


def MergeDict(a, b, path=None, allow_override=False):
    "Merges dict b into dict a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                MergeDict(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            elif allow_override:
                a[key] = b[key]
            else:
                # ruff: noqa: UP031
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
