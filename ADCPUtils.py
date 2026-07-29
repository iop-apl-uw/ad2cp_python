#! /usr/bin/env python
# -*- python-fmt -*-
## Copyright (c) 2023, 2024, 2025, 2026  University of Washington.
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

"""ADCPUtils.py - Utility functions."""

import argparse
import pathlib
import re
import sys
import warnings
from typing import Any, Literal

import netCDF4
import numpy as np
import scipy
import xarray as xr
from numpy.typing import NDArray

import ADCPConfig

if "BaseLog" in sys.modules:
    from BaseLog import log_critical, log_error, log_info, log_warning  # ty: ignore[unresolved-import]
else:
    from ADCPLog import log_critical, log_error, log_info, log_warning

required_python_version = (3, 10, 9)
required_numpy_version = "1.26.0"
required_scipy_version = "1.14.0"


def normalize_version(v: str | float) -> list[int]:
    """Normalizes version stamps.

    Args:
        v: Version string (or number), e.g. ``"1.2.0"``. Very old versions of
            base_station_version, for example, were stored as floats.

    Returns:
        Version as a list of integer components, e.g. ``[1, 2]`` (trailing
        ``.0`` components are dropped).
    """
    if not isinstance(v, str):
        v = str(v)  # very old versions of base_station_version for example were stored as floats
    return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]


# TODO - when python version is greater then 3.11, go to this approach to report the version
# import tomllib
# from pathlib import Path

# # Adjust path to find your pyproject.toml relative to the script
# path = Path(__file__).parent / "pyproject.toml"

# with open(path, "rb") as f:
#     data = tomllib.load(f)

# # Structure depends on your build backend (Poetry vs. Standard PEP 621)
# version = data.get("project", {}).get("version") or \
#           data.get("tool", {}).get("poetry", {}).get("version")

# print(f"Local version: {version}")


def check_versions() -> int:
    """Checks for minimum versions of critical libraries and python.

    Returns:
        0 for no issue, 1 for version outside minimum
    """
    log_info("ADCP version: 0.0.3")

    # TODO - get this working
    # Try for the commit id
    # cmd_line = f"git --git-dir {pathlib.Path(base_opts.basestation_directory) / '.git'} describe --always"

    # try:
    #     sts, fo = run_cmd_shell(cmd_line)
    #     if not sts and fo is not None:
    #         log_info(f"Commit-ID: {fo.readline().decode().rstrip()}")
    #         fo.close()
    #     else:
    #         log_info("Commit-ID: unknown")
    # except Exception:
    #     log_info("Commit-ID: unknown")

    # Check python version
    log_info(f"{sys.version_info[0]:d}.{sys.version_info[1]:d}.{sys.version_info[2]:d}")

    # Python version
    if sys.version_info < (3, 10):
        log_error("Python version 3.10 at minimum is required")
        return 1

    # Check numpy version
    if normalize_version(np.__version__) < normalize_version(required_numpy_version):
        log_critical(f"Numpy {required_numpy_version} or greater required - current version {np.__version__}")
        return 1

    # Check scipy version
    if normalize_version(scipy.__version__) < normalize_version(required_scipy_version):
        log_critical(f"Scipy {required_scipy_version} or greater required - current version {scipy.__version__}")
        return 1
    return 0


def open_netcdf_file(
    ncf_name: pathlib.Path,
    mode: Literal["r", "w", "r+", "a", "x", "rs", "ws", "r+s", "as"] = "r",
    mask_results: bool = False,
) -> None | netCDF4.Dataset:
    """Opens a netCDF file, removing any existing file first if opening for write.

    Args:
        ncf_name: Path to the netCDF file to open.
        mode: netCDF4 file mode.
        mask_results: If True, masked arrays are returned for variables with missing data.

    Returns:
        The open dataset, or None if the file could not be opened.
    """
    # netCDF4 tries to open with a write exclusive, which will fail if some other process has
    # the file open for read.
    if "w" in mode:
        try:
            ncf_name.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            log_warning(f"Failed to remove {ncf_name} before write", "exc")
    try:
        ds = netCDF4.Dataset(ncf_name, mode)
    except Exception:
        log_error(f"Failed to open {ncf_name}", "exc")
        return None
    ds.set_auto_mask(mask_results)
    return ds


def intnan(y: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Interpolates over NaNs.

    Leading and trailing NaNs are replaced with the first/last non-nan value.
    Works on one dimensional items only.

    Args:
        y: Array to interpolate over, updated in place.

    Returns:
        ``y``, with interior NaNs interpolated over, or None if ``y`` has no
        non-NaN values to interpolate from.
    """
    sz = y.size
    good_mask = np.logical_not(np.isnan(y))
    i_good = np.squeeze(np.nonzero(good_mask))
    if i_good.size == 0:
        log_error("No good points - cannot interpolate over nans")
        return None
    # Fill in the tails
    y[0 : np.min(i_good)] = y[np.min(i_good)]
    y[np.max(i_good) + 1 :] = y[np.max(i_good)]
    # Regen mask now that the tails are filled in
    good_mask = np.logical_not(np.isnan(y))
    i_good = np.nonzero(good_mask)
    t = np.arange(sz)
    f = scipy.interpolate.interp1d(
        t[good_mask],
        y[good_mask],
        # bounds_error=False,
        # fill_value="extrapolate",
    )
    y[np.logical_not(good_mask)] = f(t[np.logical_not(good_mask)])
    return y


# function dx = lon_to_m(dlon, alat)
# % dx = lon_to_m(dlon, alat)
# % dx   = longitude difference in meters
# % dlon = longitude difference in degrees
# % alat = average latitude between the two fixes

# rlat = alat * pi/180;
# p = 111415.13 * cos(rlat) - 94.55 * cos(3 * rlat);
# dx = dlon .* p;


def lon_to_m(dlon: NDArray[np.float64], alat: np.float64) -> NDArray[np.float64]:
    """Converts a longitude difference to a distance in meters.

    Args:
        dlon: Longitude difference(s), in degrees.
        alat: Average latitude between the two fixes, in degrees.

    Returns:
        Distance(s) in meters corresponding to ``dlon`` at latitude ``alat``.
    """
    rlat = alat * np.pi / 180.0
    p = 111415.13 * np.cos(rlat) - 94.55 * np.cos(3 * rlat)
    ret_val: NDArray[np.float64] = dlon * p
    return ret_val


# function dy = lat_to_m(dlat,alat)
# %  dy = lat_to_m(dlat,alat)
# % dy   = latitude difference in meters
# % dlat = latitude difference in degrees
# % alat = average latitude between the two fixes
# % Reference: American Practical Navigator, Vol II, 1975 Edition, p 5

# rlat = alat * pi/180;
# m = 111132.09 * ones(size(rlat)) - ...
#     566.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat);
# dy = dlat .* m ;


def lat_to_m(dlat: NDArray[np.float64], alat: np.float64) -> NDArray[np.float64]:
    """Converts a latitude difference to a distance in meters.

    Reference: American Practical Navigator, Vol II, 1975 Edition, p 5.

    Args:
        dlat: Latitude difference(s), in degrees.
        alat: Average latitude between the two fixes, in degrees.

    Returns:
        Distance(s) in meters corresponding to ``dlat`` at latitude ``alat``.
    """
    rlat = alat * np.pi / 180.0
    m = 111132.09 - 566.05 * np.cos(2 * rlat) + 1.2 * np.cos(4 * rlat)
    retval: NDArray[np.float64] = dlat * m
    return retval


def latlon2xy(ll: NDArray[np.complex128], ll0: np.complex128) -> NDArray[np.complex128]:
    """Converts lon/lat positions (encoded as complex numbers) to an xy displacement in km.

    Args:
        ll: Positions as ``lon + 1j*lat``, in degrees. Updated in place (shifted by ``ll0``).
        ll0: Origin position as ``lon + 1j*lat``, in degrees.

    Returns:
        Displacement from ``ll0`` to each of ``ll``, as ``x + 1j*y`` in kilometers.
    """
    ll -= ll0

    # xy =      lon_to_m(real(ll),imag(ll0))+...
    #         i*lat_to_m(imag(ll),imag(ll0));

    xy: NDArray[np.complex128] = lon_to_m(ll.real, ll0.imag) + 1j * lat_to_m(ll.imag, ll0.imag)
    return xy * 1e-3


def IT_sg_interp_AP(
    x: NDArray[np.float64], y: NDArray[np.complex128], xi: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Complex interpolation, where the amplitude and phase of y are linearly interpolated.

    Does not require a regular ``xi``.

    Args:
        x: Sample locations for ``y``. Only finite entries are used.
        y: Complex sample values at each ``x``.
        xi: Locations to interpolate onto.

    Returns:
        Complex-interpolated values of ``y`` at ``xi``.
    """
    # ii = find(isfinite(x));
    ii = np.nonzero(np.isfinite(x))[0]

    # % Amplitude - easy
    # if length(x(ii))==1
    #   Ai = abs(y(ii));
    # else
    #   Ai = interp1(x(ii),abs(y(ii)), xi);
    # end

    # Amplitude - easy
    if np.shape(x[ii])[0] == 1:
        Ai = np.abs(y[ii])
    else:
        f = scipy.interpolate.interp1d(
            x[ii],
            np.abs(y[ii]),
            bounds_error=False,
            # fill_value="extrapolate",
        )
        Ai = f(xi)

    # % Phase - a little more tricky
    # P = unwrap(angle(y(ii)));
    # if length(x(ii))==1
    #   Pi = P(ii);
    # else
    #   Pi = interp1(x(ii),P, xi);
    # end
    # yi = Ai.*exp(1i*Pi);

    # Phase - a little more tricky
    P = np.unwrap(np.angle(y[ii]))
    if np.shape(x[ii])[0] == 1:
        Pi = P[ii]
    else:
        f = scipy.interpolate.interp1d(
            x[ii],
            P,
            bounds_error=False,
            # fill_value="extrapolate",
        )
        Pi = f(xi)

    yi: NDArray[np.complex128] = Ai * np.exp(1j * Pi)
    return yi


# function[yi]=course_interp(x,y,xi)
# % Y is a heading in degrees.
# y0 = IT_sg_interp_AP(x,exp(1i*y*pi/180),xi);
# yi = angle(y0)*180/pi;
# yi(yi<0)=yi(yi<0)+360;


def course_interp(
    time: NDArray[np.float64], heading: NDArray[np.float64], new_time: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolates course (heading) onto a new time grid.

    Args:
        time: Sample times for ``heading``.
        heading: Heading samples, in degrees, at each ``time``.
        new_time: Times to interpolate onto.

    Returns:
        Interpolated heading, in degrees in ``[0, 360)``, at each ``new_time``.
    """
    time0 = IT_sg_interp_AP(time, np.exp(1j * heading * np.pi / 180), new_time)
    timei = np.angle(time0) * 180 / np.pi
    timei[timei < 0] = timei[timei < 0] + 360
    return timei


# Not currently used
def interp1d(
    first_epoch_time_s_v: NDArray[np.float64],
    data_v: NDArray[np.float64],
    second_epoch_time_s_v: NDArray[np.float64],
    kind: Literal[
        "linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"
    ] = "linear",
) -> NDArray[np.float64]:
    """For each data point in ``data_v`` at ``first_epoch_time_s_v``, determines the value at ``second_epoch_time_s_v``.

    Interpolates according to ``kind``. Assumes both epoch-time arrays increase
    monotonically. Extends ``first_epoch_time_s_v``/``data_v`` with 'nearest' values
    at the ends so they cover the full range of ``second_epoch_time_s_v``.

    Args:
        first_epoch_time_s_v: Sample times, in epoch seconds, for ``data_v``.
        data_v: Data samples at each ``first_epoch_time_s_v``.
        second_epoch_time_s_v: Times, in epoch seconds, to interpolate onto.
        kind: Interpolation kind, passed to ``scipy.interpolate.interp1d``.

    Returns:
        Interpolated data at each ``second_epoch_time_s_v``.
    """
    # add 'nearest' data item to the ends of data and first_epoch_time_s_v
    if second_epoch_time_s_v[0] < first_epoch_time_s_v[0]:
        # Copy the first value below the interpolation range
        data_v = np.append(np.array([data_v[0]]), data_v)
        first_epoch_time_s_v = np.append(np.array([second_epoch_time_s_v[0]]), first_epoch_time_s_v)

    if second_epoch_time_s_v[-1] > first_epoch_time_s_v[-1]:
        # Copy the last value above the interpolation range
        data_v = np.append(data_v, np.array([data_v[-1]]))
        first_epoch_time_s_v = np.append(first_epoch_time_s_v, np.array([second_epoch_time_s_v[-1]]))

    interp_data_v: NDArray[np.float64] = scipy.interpolate.interp1d(first_epoch_time_s_v, data_v, kind=kind)(
        second_epoch_time_s_v
    )
    return interp_data_v


def interp_nm(
    x: NDArray[np.float64], y: NDArray[np.float64], xi: NDArray[np.float64], fill_value: float = np.nan
) -> NDArray[np.float64] | None:
    """Interpolates a non-monotonic x along a monotonic xi.

    Args:
        x: Sample locations (non-monotonic), paired with ``y``.
        y: Sample values at each ``x``.
        xi: Monotonic locations to interpolate onto.
        fill_value: Value used for ``xi`` outside the range of (sorted) ``x``.

    Returns:
        Interpolated values of ``y`` at ``xi``, or None if the fit failed to
        converge after too many iterations.
    """
    index = np.nonzero(np.logical_and(np.isfinite(x), np.isfinite(y)))[0]
    if np.shape(index)[0] <= 2:
        return np.zeros(np.shape(xi)[0]) * np.nan
    P = np.polyfit(x[index] - x[index[0]], y[index], 1)
    flag_sign = int(np.sign(P[0]))
    if flag_sign == 0:
        retval: NDArray[np.float64] = np.ones(np.shape(xi)[0]) * P[1]
        return retval

    # Code for multi-dimension - not handled here
    # [m n]=size(x);
    # if n>m
    #     Y= sortrows([transpose(x(index))  transpose(y(index)) ] , 1*flag_sign);
    # else
    #     Y= sortrows([x(index)  y(index) ] , 1*flag_sign);
    # end

    indicies = np.argsort(x)[::flag_sign]
    Y = np.vstack((x[index], y[index])).T[indicies, :]

    dy = np.nanmedian(np.diff(Y[:, 0]))
    if np.abs(dy) < 1e-5:
        dy = np.nanmean(np.diff(Y[:, 0]))

    kk = 0
    max_k = 1e4
    while kk < max_k:
        indicies = np.nonzero(np.diff(Y[:, 0]) * flag_sign <= 0)[0] + 1
        Y[indicies, 0] = Y[indicies, 0] + dy * 0.0001
        kk += 1
        if len(np.nonzero(Y[indicies, 0])[0]) == 0:
            break

    if kk >= max_k:
        log_error("Could not interpolate", "exc")
        return None

    # yi=interp1(Y(:,1), Y(:,2), xi);
    retval2: NDArray[np.float64] = scipy.interpolate.interp1d(
        Y[:, 0],
        Y[:, 1],
        bounds_error=False,
        fill_value=fill_value,
    )(xi)
    return retval2


# function[in]=find_nearest(x,y)
# % for each element of x, find the closest to y (???)

# in = zeros(size(y))*NaN;
# for n=1:length(in)
#   [~,in(n)]=min(abs(x-y(n)));
# end


# From https://stackoverflow.com/questions/40890960/numpy-scipy-equivalent-of-matlabs-sparse-function
def sparse(
    i: NDArray[np.int_], j: NDArray[np.int_], v: NDArray[np.float64], m: int, n: int
) -> scipy.sparse.csr_array:
    """Creates a compressed sparse matrix that has many zeros.

    Args:
        i: 1-D array of index-1 values, size n1.
        j: 1-D array of index-2 values, size n1.
        v: 1-D array of values, size n1.
        m: x size of the matrix, >= n1.
        n: y size of the matrix, >= n1.

    Returns:
        2-D sparse matrix, full of zeros excepting values ``v`` at indexes ``(i, j)``.
    """
    return scipy.sparse.csr_array((v, (i, j)), shape=(m, n))


def bindata(
    x: NDArray[np.float64], y: NDArray[np.float64], gx: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bins y(x) onto b(gx), gx defining centers of the bins. NaNs ignored.

    Note: current implementation only handles the 1-D case.

    Args:
        x: Sample locations, paired with ``y``.
        y: Sample values at each ``x``. NaNs are ignored.
        gx: Bin center locations.

    Returns:
        A tuple ``(b, n)``:
            b: Binned (averaged) data, one value per bin in ``gx``.
            n: Number of points in each bin.
    """
    # x=x(:);y=y(:);
    x = np.atleast_1d(np.squeeze(x))
    y = np.atleast_1d(np.squeeze(y))

    # idx=find(~isnan(y));
    # if (isempty(idx)) b=nan*gx;n=nan*gx;s=nan*gx;return;end
    idx = np.nonzero(np.logical_not(np.isnan(y)))[0]
    if idx.size == 0:
        return (np.nan * gx, np.nan * gx)

    # x=x(idx);
    # y=y(idx);
    x = x[idx]
    y = y[idx]

    # xx = gx(:)';
    # binwidth = diff(xx) ;
    # xx = [xx(1)-binwidth(1)/2 xx(1:end-1)+binwidth/2 xx(end)+binwidth(end)/2];
    xx = np.squeeze(gx)
    binwidth = np.diff(xx)
    xx = np.hstack((xx[0] - binwidth[0] / 2, xx[0:-1] + binwidth / 2, xx[-1] + binwidth[-1] / 2))

    # % Shift bins so the interval is "( ]" instead of "[ )".
    # bins = xx + max(eps,eps*abs(xx));
    bins = xx + np.maximum(np.spacing(1) + np.zeros(xx.shape), np.spacing(1) * np.abs(xx))

    # [nn,bin] = histc(x,bins,1);
    # nn=nn(1:end-1);
    # nn(nn==0)=NaN;
    bin = np.digitize(x, bins)
    nn = np.zeros(bins.shape)
    for bb in bin:
        nn[bb - 1] += 1
    nn = nn[0:-1]
    nn[nn == 0] = np.nan

    # iidx=find(bin>0);
    # sum=full(sparse(bin(iidx),iidx*0+1,y(iidx)));
    # if length(gx)-length(sum)>0
    #   sum=[sum;zeros(length(gx)-length(sum),1)*nan];% add zeroes to the end
    # else
    #   sum=sum(1:length(nn));
    # end
    # b=sum./nn;

    iidx = np.nonzero(bin > 0)[0]
    # ii = bin[iidx]
    # jj = np.ones(iidx.shape[0])
    # mm = np.max(bin[iidx])
    # nn = 1
    # sum = sparse(ii, jj, y[iidx], mm, nn).todense()
    sum = np.zeros(np.max(bin[iidx]), dtype=y.dtype)
    for ii in iidx:
        sum[bin[ii] - 1] += y[ii]

    if np.shape(gx)[0] - np.shape(sum)[0] > 0:
        # add zeros to the end
        sum = np.hstack((sum, np.zeros(np.shape(gx)[0] - np.shape(sum)[0], dtype=y.dtype)))
    else:
        sum = sum[: np.shape(nn)[0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b = sum / nn
    return (b, nn)


def StripVars(dsi: netCDF4.Dataset, dso: netCDF4.Dataset, var_meta: dict[str, ADCPConfig.NCVarMeta]) -> None:
    """Copies dsi to dso, excluding any variables in the var_meta dict.

    Note: at this time, the dimensions are not addressed.

    Args:
        dsi: Input dataset to copy from.
        dso: Output dataset to copy into, updated in place.
        var_meta: Mapping of netCDF variable name to metadata; variables named
            here are excluded from the copy (along with any dimensions unique to them).
    """
    strip_names = [var_meta[k].nc_varname for k in var_meta]

    strip_dims = []
    for name, var in dsi.variables.items():
        if name in strip_names:
            for d in var.dimensions:
                strip_dims.append(d)

    # Check for any dims that a shared with non-strip items
    for name, var in dsi.variables.items():
        if name not in strip_names:
            for d in var.dimensions:
                if d in strip_dims:
                    strip_dims.remove(d)

    for name, dimension in dsi.dimensions.items():
        if name not in strip_dims:
            dso.createDimension(name, dimension.size)

    for name, var in dsi.variables.items():
        if name not in strip_names:
            fv = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            nc_var = dso.createVariable(
                name,
                var.datatype,
                var.dimensions,
                fill_value=fv,
                compression="zlib",
                complevel=9,
            )
            # netCDF4 is calling shape on the numpy array - which has been deprecated
            # TODO - this may be fixed wit https://github.com/Unidata/netcdf4-python/pull/1469, once
            # it gets out into release            with warnings.catch_warnings():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                nc_var[:] = var[:]
            for a in var.ncattrs():
                if a != "_FillValue":
                    nc_var.setncattr(a, var.getncattr(a))

    for a in dsi.ncattrs():
        dso.setncattr(a, dsi.getncattr(a))


def CreateNCVar(
    dso: netCDF4.Dataset,
    template: dict[str, ADCPConfig.NCVarMeta],
    key_name: str,
    data: Any,  # noqa: ANN401 -- genuinely dynamic: str, scalar, or ndarray depending on key_name's nc_type
) -> None:
    """Creates a netCDF variable in dso and sets its metadata, from template.

    Args:
        dso: Output dataset, updated in place.
        template: Mapping of variable name to metadata.
        key_name: Name of the variable as it appears in ``template``.
        data: Input data for the variable -- a string, scalar, or ndarray,
            depending on ``template[key_name].nc_type``/``nc_dimensions``.
    """
    is_str = False
    if isinstance(data, str):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            inp_data = netCDF4.stringtochar(np.array(data, dtype=f"S{len(data)}"), encoding="ascii").squeeze()
        is_str = True
    elif np.ndim(data) == 0:
        # Scalar data
        inp_data = np.dtype(template[key_name].nc_type).type(data)
    else:
        inp_data = data.astype(template[key_name].nc_type)

    if hasattr(template[key_name], "decimal_pts") and template[key_name].decimal_pts >= 0:
        inp_data = inp_data.round(template[key_name].decimal_pts)

    # Check for scalar variables
    if np.ndim(inp_data) == 0:
        if inp_data == np.nan:
            inp_data = template[key_name].nc_attribs.FillValue
    elif not is_str:
        inp_data[np.isnan(inp_data)] = template[key_name].nc_attribs.FillValue

    # assert len(template[key_name]["nc_dimensions"]) == 1
    # if template[key_name]["nc_dimensions"][0] not in dso.dimensions:
    #    dso.createDimension(template[key_name]["nc_dimensions"][0], np.shape(data)[0])

    for ii, dim in enumerate(template[key_name].nc_dimensions):
        if dim not in dso.dimensions:
            if is_str:
                dso.createDimension(dim, np.shape(inp_data)[ii])
            else:
                dso.createDimension(dim, np.shape(data)[ii])

    nc_var = dso.createVariable(
        template[key_name].nc_varname,
        template[key_name].nc_type,
        template[key_name].nc_dimensions,
        fill_value=template[key_name].nc_attribs.FillValue,
        compression="zlib",
        complevel=9,
    )
    # netCDF4 is calling shape on the numpy array - which has been deprecated
    # TODO - this may be fixed wit https://github.com/Unidata/netcdf4-python/pull/1469, once
    # it gets out into release
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        nc_var[:] = inp_data
    for a_name, attrib in iter(template[key_name].nc_attribs):
        # TODO - finish this off
        # if a_name in ("valid_min", "valid_max"):
        #    nc_var.setncattr(a_name, np.array(attrib).astype(template[key_name]["nc_type"])
        if a_name != "FillValue" and attrib is not None:
            nc_var.setncattr(a_name, attrib)


def CreateNCVars(
    dso: netCDF4.Dataset, ad2cp_var_map: dict[str, Any], var_meta: dict[str, ADCPConfig.NCVarMeta]
) -> None:
    """Creates a netCDF variable for each entry in ad2cp_var_map.

    Args:
        dso: Output dataset, updated in place.
        ad2cp_var_map: Mapping of variable name to its data (scalar or ndarray).
        var_meta: Mapping of variable name to metadata.
    """
    for key_name, data in ad2cp_var_map.items():
        CreateNCVar(dso, var_meta, key_name, data)


def WriteParamsWeights(
    dso: netCDF4.Dataset,
    weights: ADCPConfig.Weights,
    params: ADCPConfig.Params,
    template: dict[str, ADCPConfig.NCVarMeta],
) -> None:
    """Writes the inverse weights and processing parameters as netCDF variables.

    Args:
        dso: Output dataset, updated in place.
        weights: Inverse weighting configuration to write out.
        params: Inverse processing parameters to write out.
        template: Mapping of variable name to metadata.
    """
    for key_name, data in weights.items():
        CreateNCVar(dso, template, key_name, data)
    for key_name, data in params.items():
        CreateNCVar(dso, template, key_name, data)


def GetMissionStr(dive_nc_file: netCDF4.Dataset) -> str:
    """Assembles the mission string from a dive netCDF file's log_ID and mission title.

    Args:
        dive_nc_file: Open dive netCDF dataset to read ``log_ID``/``sg_cal_mission_title`` from.

    Returns:
        Mission string, e.g. ``"sg171 EKAMSAT_Apr24"``.
    """
    log_id = None
    mission_title = ""
    if "log_ID" in dive_nc_file.variables:
        log_id = int(dive_nc_file.variables["log_ID"].getValue())
    if "sg_cal_mission_title" in dive_nc_file.variables:
        mission_title = dive_nc_file.variables["sg_cal_mission_title"][:].tobytes().decode("utf-8")
    # return f"sg{'%03d' % (log_id if log_id else 0,)} {mission_title}"
    return f"sg{log_id if log_id else 0:03d} {mission_title}"


def SetupPlotDirectory(adcp_opts: argparse.Namespace) -> int:
    """Ensures plot_directory exists, creating it if needed.

    Args:
        adcp_opts: Command line options; uses/updates ``adcp_opts.plot_directory``.

    Returns:
        0 for success, 1 for failure.
    """
    if adcp_opts.plot_directory.exists():
        if adcp_opts.plot_directory.is_dir():
            return 0
        else:
            log_error(f"{adcp_opts.plot_directory} is not a directory")
            return 1
    else:
        try:
            adcp_opts.plot_directory.mkdir(mode=0o775)
        except Exception:
            log_error(f"Could not create {adcp_opts.plot_directory}", "exc")
            return 1
    return 0


def isoSurface(field: xr.DataArray, target: float, dim: str) -> xr.DataArray:
    """Linearly interpolates a coordinate isosurface where a field equals a target.

    Args:
        field: The field in which to interpolate the target isosurface.
        target: The target isosurface value.
        dim: The field dimension to interpolate.

    Returns:
        The ``dim`` coordinate value(s) where ``field`` crosses ``target``.

    Examples:
        Calculate the depth of an isotherm with a value of 5.5:

        >>> temp = xr.DataArray(
        ...     range(10,0,-1),
        ...     coords={"depth": range(10)}
        ... )
        >>> isoSurface(temp, 5.5, dim="depth")
        <xarray.DataArray ()>
        array(4.5)
    """
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = field.isel(slice0).drop_vars(dim)
    field1 = field.isel(slice1).drop_vars(dim)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(crossing_mask_decr | crossing_mask_incr, 1, np.nan)

    coords0 = crossing_mask * field[dim].isel(slice0).drop_vars(dim)
    coords1 = crossing_mask * field[dim].isel(slice1).drop_vars(dim)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = coords0 + (target - field0) * (coords1 - coords0) / (field1 - field0)

    return iso.max(dim, skipna=True)
