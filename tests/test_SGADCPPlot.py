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
import shutil
from typing import cast

import netCDF4
import plotly.graph_objects
import pytest

import ADCPUtils
import SGADCP
import SGADCPPlot


class _RaisingGetncattrDataset:
    """Proxy forwarding to a real Dataset but with getncattr made to raise.

    netCDF4.Dataset is a compiled extension type; on some platform builds it
    is an immutable type that rejects monkeypatching class attributes, so this
    wraps an instance instead of patching netCDF4.Dataset directly.
    """

    def __init__(self, wrapped: netCDF4.Dataset) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)

    def getncattr(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")


def _single_dive_mission(tmp_path: pathlib.Path) -> pathlib.Path:
    mission_dir = tmp_path / "mission"
    mission_dir.mkdir()
    shutil.copy("testdata/downward/p2650001.nc", mission_dir / "p2650001.nc")
    return mission_dir


def _build_output_nc(tmp_path: pathlib.Path) -> pathlib.Path:
    mission_dir = _single_dive_mission(tmp_path)
    assert SGADCP.main(["--mission_dir", str(mission_dir)]) == 0
    (output_nc,) = mission_dir.glob("*-adcp-realtime.nc")
    return output_nc


def test_main_happy_path_writes_html_and_webp(tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    plots_dir = tmp_path / "plots"
    result = SGADCPPlot.main([str(output_nc), "--plot_directory", str(plots_dir)])
    assert result == 0
    (html_file,) = plots_dir.glob("*.html")
    (webp_file,) = plots_dir.glob("*.webp")
    assert html_file.stat().st_size > 0
    assert webp_file.stat().st_size > 0


def test_main_default_plot_directory(tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    result = SGADCPPlot.main([str(output_nc)])
    assert result == 0
    default_plots_dir = output_nc.parent / "plots"
    assert default_plots_dir.is_dir()
    assert list(default_plots_dir.glob("*.html"))


def test_main_adcp_opts_none_returns_1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(SGADCPPlot.ADCPOpts, "ADCPOptions", lambda *_a, **_k: None)
    assert SGADCPPlot.main(["unused.nc"]) == 1


def test_main_check_versions_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    monkeypatch.setattr(SGADCPPlot.ADCPUtils, "check_versions", lambda: 1)
    assert SGADCPPlot.main([str(output_nc), "--plot_directory", str(tmp_path / "plots")]) == 1


def test_main_setup_plot_directory_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    monkeypatch.setattr(SGADCPPlot.ADCPUtils, "SetupPlotDirectory", lambda *_a, **_k: 1)
    assert SGADCPPlot.main([str(output_nc), "--plot_directory", str(tmp_path / "plots")]) == 1


def test_generate_plots_ds_none_returns_none(tmp_path: pathlib.Path):
    bad_nc = tmp_path / "bad.nc"
    bad_nc.write_text("not a real netcdf file")
    opts = argparse.Namespace(min_plot_depth=0.0, max_plot_depth=1000.0)
    assert SGADCPPlot.GeneratePlots(bad_nc, opts) is None


def test_generate_plots_min_depth_greater_than_max_returns_none(tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    opts = argparse.Namespace(min_plot_depth=1000.0, max_plot_depth=0.0)
    assert SGADCPPlot.GeneratePlots(output_nc, opts) is None


def test_generate_plots_plot_func_exception_logged(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(SGADCPPlot, "PlotOceanVelocity", _raise)
    opts = argparse.Namespace(min_plot_depth=0.0, max_plot_depth=1000.0)
    # Exception is caught and logged inside GeneratePlots - no raise expected.
    assert SGADCPPlot.GeneratePlots(output_nc, opts) is None


def test_plot_ocean_velocity_missing_variable_returns_none(tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    # Remove a required variable so PlotOceanVelocity's KeyError branch fires.
    with netCDF4.Dataset(output_nc, "r+") as ds:
        ds.renameVariable("ad2cp_inv_profile_uocn", "renamed_uocn")
    ds = ADCPUtils.open_netcdf_file(output_nc, mask_results=True)
    assert ds is not None
    opts = argparse.Namespace(min_plot_depth=0.0, max_plot_depth=1000.0)
    assert SGADCPPlot.PlotOceanVelocity(output_nc, ds, opts) is None
    ds.close()


def test_plot_ocean_velocity_generic_exception_returns_none(tmp_path: pathlib.Path):
    output_nc = _build_output_nc(tmp_path)
    ds = ADCPUtils.open_netcdf_file(output_nc, mask_results=True)
    assert ds is not None

    # getncattr is the last statement in PlotOceanVelocity's variable-reading
    # try block, so raising here (a non-KeyError) hits the generic
    # "except Exception:" branch specifically, distinct from the KeyError one.
    opts = argparse.Namespace(min_plot_depth=0.0, max_plot_depth=1000.0)
    wrapped_ds = cast(netCDF4.Dataset, _RaisingGetncattrDataset(ds))
    assert SGADCPPlot.PlotOceanVelocity(output_nc, wrapped_ds, opts) is None
    ds.close()


def test_write_output_files_no_plot_directory_returns_empty_list():
    opts = argparse.Namespace(plot_directory=None)
    fig = plotly.graph_objects.Figure()
    assert SGADCPPlot.write_output_files(opts, "base_name", fig) == []
