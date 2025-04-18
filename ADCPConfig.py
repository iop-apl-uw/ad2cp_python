#! /usr/bin/env python
# -*- python-fmt -*-
## Copyright (c) 2023, 2024, 2025  University of Washington.
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

# import pdb
import pathlib
import sys
from dataclasses import field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from pydantic.dataclasses import dataclass

import ExtendedDataClass

if "BaseLog" in sys.modules:
    from BaseLog import log_debug, log_error
else:
    from ADCPLog import log_debug, log_error


@dataclass(config=dict(extra="forbid", arbitrary_types_allowed=True))
class Params(ExtendedDataClass.ExtendedDataClass):
    # Gilder number int
    sg_id: int = field(default=0)
    # Size of vertical grid bins used in inverse solution
    dz: float = field(default=5.0)
    # Max depth for vertical grid bins used innverse solution
    depth_max: float = field(default=1000.0)
    # surface blank (if W_SURFACE ~=0, glider velocity is zero AND ADCP data is ignored above that depth)
    sfc_blank: float = field(default=3.0)
    # Which bins to use in the solution
    index_bins: npt.ArrayLike = field(default_factory=(lambda: np.arange(15)))
    # 'gsm' or 'FlightModel'
    VEHICLE_MODEL: str = field(default="FlightModel")
    # Use the glider pressure sensor
    use_glider_pressure: bool = field(default=True)
    # maximum tilt (*** probably not correct, since Nortek is using a 3-beam
    # solution with different beam angles. It assumes a non-zero pitch).
    MAX_TILT: float = field(default=0.7)  # 0.7 instead of typical 0.8. (0.7 is about a 45+17.4 degree pitch!)
    # difference between vertical velocity measured by ADCP and vertical motion of the glider.
    WMAX_error: float = field(default=0.05)
    # maximum relative velocity
    UMAX: float = field(default=0.5)
    # Is the ADCP mounted upward or downward looking
    up_looking: bool = field(default=True)

    # Min and max time for each data set
    time_limits: npt.NDArray[np.float64] = field(default_factory=(lambda: np.empty(0)))

    def __post_init__(self) -> None:
        # TODO - short term hack for new input from the config yaml.  Right now, these
        # are not converted to the correct type by pydantic.
        try:
            if not isinstance(self.time_limits, np.ndarray):
                self.time_limits = np.array(self.time_limits)
        except Exception:
            log_error('Failed to convert "time_limits"', "exc")
        try:
            if not isinstance(self.index_bins, np.ndarray):
                self.index_bins = np.array(self.index_bins, np.int32)
        except Exception:
            log_error('Failed to convert "index_bins"', "exc")


@dataclass(config=dict(extra="forbid"))
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


@dataclass(config=dict(extra="forbid"))
class ConfigModel:
    weights: Weights | None
    params: Params | None


def ProcessConfigFile(config_file_name: pathlib.PosixPath) -> tuple[Params, Weights] | tuple[None, None]:
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

    for k in ("weights", "params"):
        if k not in cfg_dict:
            cfg_dict[k] = {}

    try:
        config_model = ConfigModel(**cfg_dict)
    except ValidationError as e:
        for error in e.errors():
            location = f"{':'.join([str(x) for x in error['loc']])}"
            log_error(f"In {config_file_name} - {location}, {error['msg']}")
        return (None, None)

    log_debug(config_model)

    if config_model.params is None or config_model.weights is None:
        return (None, None)

    return (config_model.params, config_model.weights)


# class AttributeDict(dict[Any, Any]):
#     """Allow dot access for dictionaries"""

#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setitem__  # type: ignore
#     __delattr__ = dict.__delitem__  # type: ignore


# class NCDataType(enum.Enum):
#     f: str = "f"
#     d: str = "d"
#     i: str = "i"


# Models for var_meta.yml file contents
# class NCCoverageContentType(enum.Enum):
#    physicalMeasurement: str = "physicalMeasurement"
#    coordinate: str = "coordinate"
#    modelResult: str = "modelResult"
#    auxiliaryInformation: str = "auxiliaryInformation"


class NCAttribs(BaseModel):
    FillValue: StrictFloat
    description: StrictStr
    units: StrictStr
    # coverage_content_type: NCCoverageContentType
    coverage_content_type: Literal["physicalMeasurement", "coordinate", "modelResult", "auxiliaryInformation"]
    comments: StrictStr | None = None
    standard_name: StrictStr | None = None
    model_config = ConfigDict(extra="allow")


class NCVarMeta(BaseModel):
    nc_varname: StrictStr
    nc_dimensions: list[StrictStr]
    nc_attribs: NCAttribs
    nc_type: Literal["f", "d", "i", "c"]
    decimal_pts: StrictInt
    model_config = ConfigDict(extra="forbid")


def LoadVarMeta(var_meta_file: pathlib.Path) -> dict[str, NCVarMeta]:
    """Loads and validates yaml data"""
    var_meta = {}
    try:
        with open(var_meta_file, "r") as fi:
            var_meta_tmp = yaml.safe_load(fi)

        for k, v in var_meta_tmp.items():
            try:
                if isinstance(v, dict):
                    m = NCVarMeta(**v)
                else:
                    log_error(f"{k} in {var_meta_file} is not a dictionary")
                    continue
            except ValidationError as e:
                for error in e.errors():
                    # pdb.set_trace()
                    location = f"{k}:{':'.join([str(x) for x in error['loc']])}"
                    log_error(f"In {var_meta_file} - {location}, {error['msg']}")
                log_error(f"Skipping {k}")
            else:
                # var_meta[k] = AttributeDict(var_meta_tmp[k])
                var_meta[k] = m
    except Exception:
        log_error(f"Could not process {var_meta_file}", "exc")

    return var_meta


def LoadGlobalMeta(global_meta_file_local: pathlib.Path) -> dict[str, Any]:
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


def MergeDict(
    a: dict[Any, Any], b: dict[Any, Any], path: list[str] | None = None, allow_override: bool = False
) -> dict[Any, Any]:
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
