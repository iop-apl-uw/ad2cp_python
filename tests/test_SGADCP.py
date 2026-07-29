# -*- python-fmt -*-

## Copyright (c) 2024  University of Washington.
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
import shutil
from typing import Literal

import netCDF4
import pytest

import ADCPUtils
import SGADCP

downward_dir = "testdata/downward"
upward_dir = "testdata/upward"
cmd_lines = [["--verbose", "--mission_dir", downward_dir], ["--verbose", "--mission_dir", upward_dir]]


@pytest.mark.parametrize("cmd_line", cmd_lines)
def test_downward(caplog: pytest.LogCaptureFixture, cmd_line: list[str]):
    result = SGADCP.main(cmd_line)
    assert result == 0
    for record in caplog.records:
        assert record.levelname not in ["CRITICAL", "ERROR", "WARNING"]


def _single_dive_mission(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copies one real dive netCDF into a fresh mission dir, for fast single-dive runs."""
    mission_dir = tmp_path / "mission"
    mission_dir.mkdir()
    shutil.copy("testdata/downward/p2650001.nc", mission_dir / "p2650001.nc")
    return mission_dir


def test_main_explicit_ncf_files_list(tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    result = SGADCP.main(["--mission_dir", str(mission_dir), "p2650001.nc"])
    assert result == 0


def test_main_open_netcdf_file_returns_none_skips_file(tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    (mission_dir / "p2650099.nc").write_text("not a real netcdf file")
    result = SGADCP.main(["--mission_dir", str(mission_dir)])
    # The corrupt file is skipped (open_netcdf_file returns None); the other
    # real dive still processes successfully.
    assert result == 0


def test_main_unknown_calling_module_returns_1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(SGADCP.ADCPOpts, "ADCPOptions", lambda *_a, **_k: None)
    assert SGADCP.main(["--mission_dir", "unused"]) == 1


def test_main_check_versions_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    monkeypatch.setattr(SGADCP.ADCPUtils, "check_versions", lambda: 1)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_bad_config_returns_1(tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    bad_cfg = tmp_path / "bad.yml"
    bad_cfg.write_text("params:\n  bogus_key: 1\n")
    assert SGADCP.main(["--mission_dir", str(mission_dir), "--adcp_config_file", str(bad_cfg)]) == 1


def test_main_no_matching_ncf_files_returns_1(tmp_path: pathlib.Path):
    empty_mission_dir = tmp_path / "empty_mission"
    empty_mission_dir.mkdir()
    assert SGADCP.main(["--mission_dir", str(empty_mission_dir)]) == 1


@pytest.mark.parametrize(
    "target,side_effect,expect_log",
    [
        ("ADCPFiles", "ADCPReadSGNCF", "skipping"),
        ("ADCPRealtime", "TransformToInstrument", "Problem transforming compass data"),
        ("ADCP", "CleanADCP", "Problem cleaning realtime adcp data"),
        ("ADCP", "Inverse", "Problem performing inverse calculation"),
    ],
    ids=["read_sgncf_raises", "transform_raises", "clean_adcp_raises", "inverse_raises"],
)
def test_main_collaborator_exception_skips_dive_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, target: str, side_effect: str, expect_log: str
):
    mission_dir = _single_dive_mission(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(getattr(SGADCP, target), side_effect, _raise)
    result = SGADCP.main(["--mission_dir", str(mission_dir)])
    # Only dive in the mission fails, so no dives are processed at all.
    assert result == 1


def test_main_read_sgncf_keyerror_skips_dive_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise KeyError("some_var")

    monkeypatch.setattr(SGADCP.ADCPFiles, "ADCPReadSGNCF", _raise)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_clean_adcp_returns_none_skips_dive_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    monkeypatch.setattr(SGADCP.ADCP, "CleanADCP", lambda *_a, **_k: None)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_include_glider_vars_writes_glider_vars(tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    result = SGADCP.main(["--mission_dir", str(mission_dir), "--include_glider_vars"])
    assert result == 0
    (output_nc,) = mission_dir.glob("*-adcp-realtime.nc")
    with netCDF4.Dataset(output_nc) as ds:
        assert "glider_longitude" in ds.variables


def test_main_save_details_writes_hdf5_sidecar(tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    result = SGADCP.main(["--mission_dir", str(mission_dir), "--save_details"])
    assert result == 0
    assert (mission_dir / "p2650001.hdf5").exists()


def test_main_output_open_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    original_open = ADCPUtils.open_netcdf_file

    def _fail_on_write(
        ncf_name: pathlib.Path,
        mode: Literal["r", "w", "r+", "a", "x", "rs", "ws", "r+s", "as"] = "r",
        mask_results: bool = False,
    ) -> netCDF4.Dataset | None:
        if "w" in mode:
            return None
        return original_open(ncf_name, mode, mask_results)

    monkeypatch.setattr(SGADCP.ADCPUtils, "open_netcdf_file", _fail_on_write)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_create_ncvars_exception_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(SGADCP.ADCPUtils, "CreateNCVars", _raise)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_write_params_weights_exception_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(SGADCP.ADCPUtils, "WriteParamsWeights", _raise)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1


def test_main_global_attr_exception_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    mission_dir = _single_dive_mission(tmp_path)
    monkeypatch.setattr(SGADCP.ADCPConfig, "LoadGlobalMeta", lambda *_a, **_k: {})
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 1
