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

import argparse
import pathlib

import netCDF4
import numpy as np
import pytest
import xarray as xr

import ADCPConfig
import ADCPUtils


def test_normalize_version_string():
    assert ADCPUtils.normalize_version("1.2.0") == [1, 2]


def test_normalize_version_non_string_coerced():
    assert ADCPUtils.normalize_version(1.20) == [1, 2]


def test_check_versions_old_python_returns_1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ADCPUtils.sys, "version_info", (3, 9, 0))
    assert ADCPUtils.check_versions() == 1


def test_check_versions_old_numpy_returns_1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ADCPUtils, "required_numpy_version", "999.0.0")
    assert ADCPUtils.check_versions() == 1


def test_check_versions_old_scipy_returns_1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ADCPUtils, "required_scipy_version", "999.0.0")
    assert ADCPUtils.check_versions() == 1


def test_check_versions_all_good_returns_0():
    assert ADCPUtils.check_versions() == 0


def test_open_netcdf_file_write_mode_missing_file_ok(tmp_path: pathlib.Path):
    ncf = tmp_path / "new.nc"
    ds = ADCPUtils.open_netcdf_file(ncf, mode="w")
    assert ds is not None
    ds.close()


def test_open_netcdf_file_write_mode_unlink_failure_logged(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    ncf = tmp_path / "new.nc"
    ncf.write_text("not a real netcdf file")

    def _raise(_self: pathlib.Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(pathlib.Path, "unlink", _raise)
    ds = ADCPUtils.open_netcdf_file(ncf, mode="w")
    # unlink failure is logged and swallowed; "w" mode still creates/truncates
    # the file via netCDF4 regardless, so the open still succeeds.
    assert ds is not None
    ds.close()


def test_open_netcdf_file_open_failure_returns_none(tmp_path: pathlib.Path):
    missing = tmp_path / "does_not_exist.nc"
    assert ADCPUtils.open_netcdf_file(missing, mode="r") is None


def test_intnan_all_nan_returns_none():
    y = np.array([np.nan, np.nan, np.nan])
    assert ADCPUtils.intnan(y) is None


def test_it_sg_interp_ap_single_finite_point():
    x = np.array([1.0, np.nan, np.nan])
    y = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    xi = np.array([1.0])
    result = ADCPUtils.IT_sg_interp_AP(x, y, xi)
    assert result.shape == (1,)
    assert np.isclose(np.abs(result[0]), np.abs(y[0]))


def test_interp_nm_too_few_points_returns_nan_array():
    x = np.array([1.0, np.nan, np.nan])
    y = np.array([1.0, np.nan, np.nan])
    xi = np.array([0.0, 1.0])
    result = ADCPUtils.interp_nm(x, y, xi)
    assert result is not None
    assert np.all(np.isnan(result))
    assert result.shape == xi.shape


def test_interp_nm_zero_slope_returns_constant(monkeypatch: pytest.MonkeyPatch):
    # A real least-squares polyfit essentially never returns an exact-zero
    # slope for floating-point data (always some residual noise), so force
    # it here to reach the flag_sign == 0 branch.
    monkeypatch.setattr(ADCPUtils.np, "polyfit", lambda *_args, **_kwargs: np.array([0.0, 5.0]))
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 5.0, 5.0])
    xi = np.array([0.5, 1.5])
    result = ADCPUtils.interp_nm(x, y, xi)
    assert result is not None
    assert np.allclose(result, 5.0)


def test_interp_nm_near_zero_dy_uses_nanmean():
    x = np.array([0.0, 1e-7, 2e-7, 3e-7])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    xi = np.array([1e-7])
    result = ADCPUtils.interp_nm(x, y, xi)
    assert result is not None


def test_interp_nm_convergence_failure_returns_none(monkeypatch: pytest.MonkeyPatch):
    # Force np.diff() to always report a sign violation at index 0, so the
    # loop's "indicies" set never empties and it runs the full max_k
    # iterations without converging.
    original_diff = ADCPUtils.np.diff

    def _always_violating_diff(a: np.ndarray) -> np.ndarray:
        result = original_diff(a)
        if result.size:
            result = result.copy()
            result[0] = 0.0
        return result

    monkeypatch.setattr(ADCPUtils.np, "diff", _always_violating_diff)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    xi = np.array([0.5, 1.5])
    assert ADCPUtils.interp_nm(x, y, xi) is None


def _make_var_meta(nc_varname: str, dims: list[str]) -> ADCPConfig.NCVarMeta:
    return ADCPConfig.NCVarMeta(
        nc_varname=nc_varname,
        nc_dimensions=dims,
        nc_attribs=ADCPConfig.NCAttribs(
            FillValue=-999.0,
            description="test var",
            units="1",
            coverage_content_type="physicalMeasurement",
        ),
        nc_type="f",
        decimal_pts=2,
    )


def test_strip_vars_drops_unique_dim_keeps_shared_dim(tmp_path: pathlib.Path):
    dsi = netCDF4.Dataset(tmp_path / "in.nc", "w")
    dsi.createDimension("d_unique", 2)
    dsi.createDimension("d_shared", 3)
    strip_a = dsi.createVariable("strip_a", "f4", ("d_unique",))
    strip_a[:] = [1.0, 2.0]
    strip_b = dsi.createVariable("strip_b", "f4", ("d_shared",))
    strip_b[:] = [1.0, 2.0, 3.0]
    keep_a = dsi.createVariable("keep_a", "f4", ("d_shared",))
    keep_a[:] = [4.0, 5.0, 6.0]
    keep_a.setncattr("units", "m")
    dsi.setncattr("history", "test history")

    var_meta = {
        "strip_a": _make_var_meta("strip_a", ["d_unique"]),
        "strip_b": _make_var_meta("strip_b", ["d_shared"]),
    }

    dso = netCDF4.Dataset(tmp_path / "out.nc", "w")
    ADCPUtils.StripVars(dsi, dso, var_meta)

    assert "d_shared" in dso.dimensions
    assert "d_unique" not in dso.dimensions
    assert "keep_a" in dso.variables
    assert "strip_a" not in dso.variables
    assert "strip_b" not in dso.variables
    assert dso.getncattr("history") == "test history"
    assert dso.variables["keep_a"].getncattr("units") == "m"

    dsi.close()
    dso.close()


def test_setup_plot_directory_existing_dir_ok(tmp_path: pathlib.Path):
    opts = argparse.Namespace(plot_directory=tmp_path)
    assert ADCPUtils.SetupPlotDirectory(opts) == 0


def test_setup_plot_directory_existing_path_not_a_dir_logged(tmp_path: pathlib.Path):
    plot_file = tmp_path / "plots"
    plot_file.write_text("not a directory")
    opts = argparse.Namespace(plot_directory=plot_file)
    assert ADCPUtils.SetupPlotDirectory(opts) == 1


def test_setup_plot_directory_creates_missing_dir(tmp_path: pathlib.Path):
    plot_dir = tmp_path / "plots"
    opts = argparse.Namespace(plot_directory=plot_dir)
    assert ADCPUtils.SetupPlotDirectory(opts) == 0
    assert plot_dir.is_dir()


def test_setup_plot_directory_mkdir_failure_logged(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    plot_dir = tmp_path / "plots"

    def _raise(_self: pathlib.Path, mode: int = 0o777) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(pathlib.Path, "mkdir", _raise)
    opts = argparse.Namespace(plot_directory=plot_dir)
    assert ADCPUtils.SetupPlotDirectory(opts) == 1


def test_isosurface_docstring_example():
    temp = xr.DataArray(range(10, 0, -1), coords={"depth": range(10)})
    result = ADCPUtils.isoSurface(temp, 5.5, dim="depth")
    assert float(result) == 4.5


def test_isosurface_no_crossing_returns_nan():
    temp = xr.DataArray(range(10), coords={"depth": range(10)})
    result = ADCPUtils.isoSurface(temp, 100.0, dim="depth")
    assert np.isnan(float(result))
