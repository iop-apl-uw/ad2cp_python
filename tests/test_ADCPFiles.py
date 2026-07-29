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
import warnings

import h5py
import netCDF4
import numpy as np
import pytest
import scipy as sp

import ADCPConfig
import ADCPFiles


def _make_ad2cp_realtime_nc(path: pathlib.Path, n_ens: int = 2, n_cells: int = 3) -> None:
    """Creates a minimal netCDF file with all required ad2cp realtime variables.

    Opens in append mode if `path` already exists, so this can be layered onto a
    glider netCDF file already written by `_make_glider_nc` without clobbering it.
    """
    ds = netCDF4.Dataset(path, "a" if path.exists() else "w")
    ds.createDimension("ensemble", n_ens)
    ds.createDimension("cell", n_cells)
    for name in ("ad2cp_pitch", "ad2cp_roll", "ad2cp_heading", "ad2cp_time"):
        v = ds.createVariable(name, "f8", ("ensemble",))
        v[:] = np.linspace(0.0, 1.0, n_ens)
    for name in ("ad2cp_velX", "ad2cp_velY", "ad2cp_velZ"):
        v = ds.createVariable(name, "f8", ("ensemble", "cell"))
        # netCDF4 internally reshapes the numpy array, deprecated as of NumPy 2.5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            v[:] = np.ones((n_ens, n_cells))
    for name in ("ad2cp_cellSize", "ad2cp_blanking", "ad2cp_pressure", "ad2cp_soundspeed"):
        v = ds.createVariable(name, "f8", ())
        v[:] = 1.0
    ds.close()


def test_fetch_var_singleton(tmp_path: pathlib.Path):
    ncf = tmp_path / "singleton.nc"
    ds = netCDF4.Dataset(ncf, "w")
    v = ds.createVariable("scalar_var", "f8", ())
    v[:] = 42.0
    assert ADCPFiles.fetch_var(v) == 42.0
    ds.close()


def test_fetch_var_array_no_fillvalue(tmp_path: pathlib.Path):
    ncf = tmp_path / "array.nc"
    ds = netCDF4.Dataset(ncf, "w")
    ds.createDimension("n", 3)
    v = ds.createVariable("array_var", "f8", ("n",))
    v[:] = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(ADCPFiles.fetch_var(v), [1.0, 2.0, 3.0])
    ds.close()


def test_fetch_var_array_with_fillvalue(tmp_path: pathlib.Path):
    ncf = tmp_path / "array_fv.nc"
    ds = netCDF4.Dataset(ncf, "w")
    ds.createDimension("n", 3)
    v = ds.createVariable("array_var", "f8", ("n",))
    v[:] = [1.0, 999.0, 3.0]
    v.setncattr("FillValue", 999.0)
    result = ADCPFiles.fetch_var(v)
    assert np.isnan(result[1])
    np.testing.assert_array_equal(result[[0, 2]], [1.0, 3.0])
    ds.close()


def test_save_to_hdf5(tmp_path: pathlib.Path):
    realtime = ADCPFiles.ADCPRealtimeData(
        U=np.array([1.0, 2.0]),
        V=sp.sparse.csr_array(np.array([[3.0, 4.0]])),
    )
    h5_path = tmp_path / "out.h5"
    with h5py.File(h5_path, "w") as hdf:
        realtime.save_to_hdf5("adcp_realtime", hdf)

    with h5py.File(h5_path, "r") as hdf:
        grp = hdf["adcp_realtime"]
        np.testing.assert_array_equal(grp["U"][:], [1.0, 2.0])
        np.testing.assert_array_equal(grp["V"][:], [[3.0, 4.0]])


def test_adcp_realtime_init_neither_time_var_raises(tmp_path: pathlib.Path):
    ncf = tmp_path / "empty.nc"
    ds = netCDF4.Dataset(ncf, "w")
    realtime = ADCPFiles.ADCPRealtimeData()
    with pytest.raises(KeyError, match="Neither ad2cp_time nor cp_time"):
        realtime.init(ds, ncf, ADCPConfig.Params())
    ds.close()


def test_adcp_realtime_init_cp_prefix_missing_required_var_raises(tmp_path: pathlib.Path):
    ncf = tmp_path / "cp_only.nc"
    ds = netCDF4.Dataset(ncf, "w")
    ds.createDimension("ensemble", 1)
    v = ds.createVariable("cp_time", "f8", ("ensemble",))
    v[:] = [0.0]
    realtime = ADCPFiles.ADCPRealtimeData()
    with pytest.raises(KeyError, match="cp_velX"):
        realtime.init(ds, ncf, ADCPConfig.Params())
    ds.close()


@pytest.mark.parametrize("up_looking,expected_sign", [(True, 1.0), (False, -1.0)])
def test_adcp_realtime_init_success(tmp_path: pathlib.Path, up_looking: bool, expected_sign: float):
    ncf = tmp_path / "full.nc"
    _make_ad2cp_realtime_nc(ncf)
    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    realtime = ADCPFiles.ADCPRealtimeData()
    params = ADCPConfig.Params(up_looking=up_looking)
    realtime.init(ds, ncf, params)
    ds.close()

    assert realtime.U.shape == (3, 2)
    expected_tilt = expected_sign * np.cos(np.pi * realtime.pitch / 180.0) * np.cos(np.pi * realtime.roll / 180.0)
    np.testing.assert_allclose(realtime.TiltFactor, expected_tilt)
    np.testing.assert_allclose(realtime.Range, np.arange(1, 4) * realtime.cellSize / 1000.0 + realtime.blanking / 100.0)


def _make_glider_nc(
    path: pathlib.Path, omit: tuple[str, ...] = (), fillvalue_var: str | None = None, dive: int = 7
) -> None:
    """Creates a minimal glider netCDF file with all SGData.load_vars() present, except any names in omit."""
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("sample", 3)
    ds.createDimension("gps", 2)
    ds.createVariable("trajectory", "i4", ()).__setitem__((), dive)

    per_sample = (
        "longitude",
        "latitude",
        "ctd_depth",
        "temperature",
        "salinity",
        "speed",
        "vert_speed",
        "speed_gsm",
        "vert_speed_gsm",
        "sound_velocity",
        "time",
        "eng_head",
    )
    for name in per_sample:
        if name in omit:
            continue
        if name == fillvalue_var:
            v = ds.createVariable(name, "f8", ("sample",), fill_value=999.0)
            v[:] = [1.0, 999.0, 1.0]
        else:
            v = ds.createVariable(name, "f8", ("sample",))
            v[:] = [1.0, 1.0, 1.0]

    if "ctd_time" not in omit:
        v = ds.createVariable("ctd_time", "f8", ("sample",))
        v[:] = [100.0, 150.0, 200.0]

    for name in ("log_gps_time", "log_gps_lat", "log_gps_lon"):
        if name in omit:
            continue
        v = ds.createVariable(name, "f8", ("gps",))
        v[:] = [50.0, 250.0] if name == "log_gps_time" else [10.0, 10.5]

    for name in ("magnetic_variation", "depth_avg_curr_east", "depth_avg_curr_north"):
        if name in omit:
            continue
        ds.createVariable(name, "f8", ()).__setitem__((), 0.0)

    ds.close()


def test_sgdata_init_missing_variable_logged_and_skipped(tmp_path: pathlib.Path):
    ncf = tmp_path / "glider.nc"
    _make_glider_nc(ncf, omit=("temperature",))
    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    glider = ADCPFiles.SGData()
    gps = ADCPFiles.GPSData()
    glider.init(ds, ncf, ADCPConfig.Params(), gps)
    ds.close()

    assert glider.temperature.size == 0


def test_sgdata_init_gsm_vehicle_model(tmp_path: pathlib.Path):
    ncf = tmp_path / "glider_gsm.nc"
    _make_glider_nc(ncf)
    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    glider = ADCPFiles.SGData()
    gps = ADCPFiles.GPSData()
    glider.init(ds, ncf, ADCPConfig.Params(VEHICLE_MODEL="gsm"), gps)
    ds.close()

    np.testing.assert_allclose(glider.Wmod, glider.vert_speed_gsm / 100.0)


def test_sgdata_init_replaces_fillvalue_with_nan(tmp_path: pathlib.Path):
    ncf = tmp_path / "glider_fv.nc"
    _make_glider_nc(ncf, fillvalue_var="salinity")
    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    glider = ADCPFiles.SGData()
    gps = ADCPFiles.GPSData()
    glider.init(ds, ncf, ADCPConfig.Params(), gps)
    ds.close()

    assert np.isnan(glider.salinity[1])
    np.testing.assert_array_equal(glider.salinity[[0, 2]], [1.0, 1.0])


def test_sgdata_init_picks_up_next_dive_gps_fix(tmp_path: pathlib.Path):
    ncf = tmp_path / "p0000007.nc"
    _make_glider_nc(ncf, dive=7)
    next_ncf = tmp_path / "p0000008.nc"
    next_ds = netCDF4.Dataset(next_ncf, "w")
    next_ds.createDimension("gps", 2)
    for name, values in (
        ("log_gps_lon", [-121.0, -120.0]),
        ("log_gps_lat", [11.0, 12.0]),
        ("log_gps_time", [260.0, 270.0]),
    ):
        v = next_ds.createVariable(name, "f8", ("gps",))
        v[:] = values
    next_ds.close()

    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    glider = ADCPFiles.SGData()
    gps = ADCPFiles.GPSData()
    glider.init(ds, ncf, ADCPConfig.Params(), gps)
    ds.close()

    # Original 2 gps fixes plus the next dive's second entry (index 1) appended
    assert gps.log_gps_time.size == 3
    assert gps.log_gps_time[-1] == 270.0
    # ADCPUtils.latlon2xy shifts gps.LL in place by the origin fix (the current dive's
    # first log_gps_lon/log_gps_lat, both 10.0 in _make_glider_nc) while computing gps.XY.
    origin = 10.0 + 1j * 10.0
    assert gps.LL[-1] == (-120.0 + 1j * 12.0) - origin


def test_adcp_read_sgncf(tmp_path: pathlib.Path):
    ncf = tmp_path / "p000007.nc"
    _make_glider_nc(ncf, dive=7)
    _make_ad2cp_realtime_nc(ncf)
    ds = netCDF4.Dataset(ncf, "r")
    ds.set_auto_mask(False)
    glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, ncf, ADCPConfig.Params())
    ds.close()

    assert isinstance(glider, ADCPFiles.SGData)
    assert isinstance(gps, ADCPFiles.GPSData)
    assert isinstance(adcp_realtime, ADCPFiles.ADCPRealtimeData)
    assert glider.dive == 7
    assert adcp_realtime.U.shape == (3, 2)
