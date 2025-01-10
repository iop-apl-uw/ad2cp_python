# -*- python-fmt -*-

## Copyright (c) 2024, 2025  University of Washington.
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

import pathlib

import numpy as np
import pytest
import testutils
import xarray as xr


def test_extension(caplog):
    """Tests directly invoking the extension to process and existing Seaglider netCDF file"""

    # A test like this could be run out of the adcp project in the sub-directory - trying for the direct load of
    # the BaseADCP extension and skipping if it doesn't load (ie. being run as a github workflow)

    BaseADCP = pytest.importorskip("BaseADCP", reason="Not installed under basestation3")

    data_dir = pathlib.Path("testdata/sg171_EKAMSAT_Apr24")
    mission_dir = data_dir.joinpath("mission_dir")

    allowed_msgs = [""]
    cmd_line = [
        "--verbose",
        "--mission_dir",
        str(mission_dir),
    ]

    testutils.run_mission(
        data_dir,
        mission_dir,
        BaseADCP.main,
        cmd_line,
        caplog,
        allowed_msgs,
    )

    # Check for variables
    dsi = xr.load_dataset(mission_dir.joinpath("p1710100.nc"))

    var_dict = {
        "ad2cp_inv_profile_uocn": np.dtype("float32"),
        "ad2cp_inv_profile_wocn": np.dtype("float32"),
        "ad2cp_velX": np.dtype("float64"),
    }

    for v, t in var_dict.items():
        assert v in dsi.variables
        assert dsi.variables[v].dtype == t
