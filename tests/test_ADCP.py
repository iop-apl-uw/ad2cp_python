# -*- python-fmt -*-

## Copyright (c) 2026  University of Washington.
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

import ADCP
import ADCPConfig
import ADCPFiles
import ADCPRealtime
import ADCPUtils

DIVE_NCF = pathlib.Path("testdata/downward/p2650001.nc")


def _load_raw() -> tuple[ADCPFiles.ADCPRealtimeData, ADCPFiles.SGData, ADCPFiles.GPSData, ADCPConfig.Params]:
    """Loads one real dive's data through the same pipeline SGADCP.main() uses.

    Real (not synthetic) fixture data, per this repo's existing test convention -
    hand-building minimal-but-valid synthetic arrays satisfying every shape/
    finiteness assumption in Inverse() would be far more fragile than reusing
    a real, already-processed dive.
    """
    param = ADCPConfig.Params()
    ds = ADCPUtils.open_netcdf_file(DIVE_NCF, mode="r")
    assert ds is not None
    glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, DIVE_NCF, param)
    param.time_limits = np.array((np.min(gps.log_gps_time), np.max(gps.log_gps_time)))
    ADCPRealtime.TransformToInstrument(adcp_realtime)
    ds.close()
    return adcp_realtime, glider, gps, param


def _load_cleaned() -> tuple[ADCPFiles.ADCPRealtimeData, ADCPFiles.GPSData, ADCPFiles.SGData, ADCPConfig.Params]:
    adcp_realtime, glider, gps, param = _load_raw()
    cleaned = ADCP.CleanADCP(adcp_realtime, glider, param)
    assert cleaned is not None
    return cleaned, gps, glider, param


def test_clean_adcp_glider_pressure_branch():
    adcp_realtime, glider, gps, param = _load_raw()
    result = ADCP.CleanADCP(adcp_realtime, glider, param)
    assert result is not None
    assert "Z" in result


def test_clean_adcp_adcp_pressure_branch():
    adcp_realtime, glider, gps, param = _load_raw()
    param.use_glider_pressure = False
    result = ADCP.CleanADCP(adcp_realtime, glider, param)
    assert result is not None
    assert "Z" in result


def test_clean_adcp_all_nan_sound_velocity_returns_none(capsys: pytest.CaptureFixture[str]):
    adcp_realtime, glider, gps, param = _load_raw()
    glider.sound_velocity = np.full_like(glider.sound_velocity, np.nan)
    result = ADCP.CleanADCP(adcp_realtime, glider, param)
    assert result is None
    assert "Failed interpolation of nans in sound velocity" in capsys.readouterr().err


def test_clean_adcp_warns_when_deeper_than_depth_max(capsys: pytest.CaptureFixture[str]):
    adcp_realtime, glider, gps, param = _load_raw()
    param.depth_max = 0.0
    result = ADCP.CleanADCP(adcp_realtime, glider, param)
    assert result is not None
    assert "deeper than ocean grid" in capsys.readouterr().err


def test_inverse_populates_inverse_tmp_debug_dict():
    cleaned, gps, glider, param = _load_cleaned()
    weights = ADCPConfig.Weights()
    inverse_tmp: dict = {}
    D, profile, plot_vars, returned_tmp = ADCP.Inverse(cleaned, gps, glider, weights, param, inverse_tmp)
    assert plot_vars is None
    assert returned_tmp is inverse_tmp
    assert "d_adcp" in inverse_tmp
    assert "Av" in inverse_tmp
    assert D.UV.shape[0] == cleaned.U.shape[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"W_SURFACE": 0},
        {"W_OCN_DNUP": 0},
        {"W_MODEL_DAC": 1},
        {"W_MODEL": 1},
        {"W_deep": 1, "W_deep_z0": 10.0},
        {"W_MODEL_bottom": True},
        {"OCN_SMOOTH": 0},
    ],
    ids=[
        "w_surface_off",
        "w_ocn_dnup_off",
        "w_model_dac_on",
        "w_model_on",
        "w_deep_on",
        "w_model_bottom_on",
        "ocn_smooth_off",
    ],
)
def test_inverse_weight_toggle_branches(overrides: dict):
    cleaned, gps, glider, param = _load_cleaned()
    weights = ADCPConfig.Weights()
    for k, v in overrides.items():
        setattr(weights, k, v)
    D, profile, _plot_vars, _inverse_tmp = ADCP.Inverse(cleaned, gps, glider, weights, param, {})
    assert D.UVttw_solution.shape[0] > 0
    assert profile.z.shape[0] > 0


def test_inverse_w_ocn_dnup_nonconvergence_raises(monkeypatch: pytest.MonkeyPatch):
    cleaned, gps, glider, param = _load_cleaned()
    weights = ADCPConfig.Weights()
    monkeypatch.setattr(ADCPUtils, "interp_nm", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="interp_nm failed to converge"):
        ADCP.Inverse(cleaned, gps, glider, weights, param, None)
