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
ADCPUtils.py - Utility functions
"""

import pathlib

# import pdb
import re
import sys
import warnings

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy
import xarray as xr

if "BaseLog" in sys.modules:
    from BaseLog import log_critical, log_error, log_warning
else:
    from ADCPLog import log_critical, log_error, log_warning

required_python_version = (3, 10, 9)
required_numpy_version = "1.26.0"
required_scipy_version = "1.14.0"


def normalize_version(v):
    """Normalizes version stamps"""
    if not isinstance(v, str):
        v = str(v)  # very old versions of base_station_version for example were stored as floats
    return [int(x) for x in re.sub(r"(\.0+)*$", "", v).split(".")]


def check_versions() -> int:
    """
    Checks for minimum versions of critical libraries and python.

    Returns:
        0 for no issue, 1 for version outside minimum
    """

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


def open_netcdf_file(ncf_name: pathlib.Path, mode: str = "r", mask_results: bool = False) -> None | netCDF4.Dataset:
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


def intnan(y: npt.NDArray[np.float64]) -> None:
    """Interpolate over NaNs.  Leading and trailing NaNs are replaced wtih the first/last
    non-nan value. Works on one dimensional items only
    """
    sz = y.size
    good_mask = np.logical_not(np.isnan(y))
    i_good = np.nonzero(good_mask)
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


# function dx = lon_to_m(dlon, alat)
# % dx = lon_to_m(dlon, alat)
# % dx   = longitude difference in meters
# % dlon = longitude difference in degrees
# % alat = average latitude between the two fixes

# rlat = alat * pi/180;
# p = 111415.13 * cos(rlat) - 94.55 * cos(3 * rlat);
# dx = dlon .* p;


def lon_to_m(dlon: npt.NDArray[np.float64], alat: np.float64) -> npt.NDArray[np.float64]:
    rlat = alat * np.pi / 180.0
    p = 111415.13 * np.cos(rlat) - 94.55 * np.cos(3 * rlat)
    return dlon * p


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


def lat_to_m(dlat: npt.NDArray[np.float64], alat: np.float64) -> npt.NDArray[np.float64]:
    rlat = alat * np.pi / 180.0
    m = 111132.09 - 566.05 * np.cos(2 * rlat) + 1.2 * np.cos(4 * rlat)
    return dlat * m


def latlon2xy(ll: npt.NDArray[np.complex128], ll0: np.complex128) -> npt.NDArray[np.complex128]:
    ll -= ll0

    # xy =      lon_to_m(real(ll),imag(ll0))+...
    #         i*lat_to_m(imag(ll),imag(ll0));

    xy = lon_to_m(ll.real, ll0.imag) + 1j * lat_to_m(ll.imag, ll0.imag)
    return xy * 1e-3


def IT_sg_interp_AP(x, y, xi):
    """complex interpolation, where the amplitude and phase of y are linearly interpolated
    does not require a regular xi.
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

    yi = Ai * np.exp(1j * Pi)
    return yi


# function[yi]=course_interp(x,y,xi)
# % Y is a heading in degrees.
# y0 = IT_sg_interp_AP(x,exp(1i*y*pi/180),xi);
# yi = angle(y0)*180/pi;
# yi(yi<0)=yi(yi<0)+360;


def course_interp(
    time: npt.NDArray[np.float64], heading: npt.NDArray[np.float64], new_time: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Interpolate course (heading) onto new time grid"""
    time0 = IT_sg_interp_AP(time, np.exp(1j * heading * np.pi / 180), new_time)
    timei = np.angle(time0) * 180 / np.pi
    timei[timei < 0] = timei[timei < 0] + 360
    return timei


# Not currently used
def interp1d(first_epoch_time_s_v, data_v, second_epoch_time_s_v, kind="linear"):
    """For each data point data_v at first_epoch_time_s_v, determine the value at second_epoch_time_s_v
    Interpolate according to type
    Assumes both epoch_time_s_v arrays increase monotonically
    Ensures first_epoch_time_s_v and data_v cover second_epoch_time_s_v using 'nearest' values
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

    interp_data_v = scipy.interpolate.interp1d(first_epoch_time_s_v, data_v, kind=kind)(second_epoch_time_s_v)
    return interp_data_v


def interp_nm(x, y, xi, fill_value=np.nan):
    """interpolate a non-monotonic x along a monotonic xi"""
    index = np.nonzero(np.logical_and(np.isfinite(x), np.isfinite(y)))[0]
    if np.shape(index)[0] <= 2:
        return np.zeros(np.shape(xi)[0]) * np.nan
    P = np.polyfit(x[index] - x[index[0]], y[index], 1)
    flag_sign = int(np.sign(P[0]))
    if flag_sign == 0:
        return np.ones(np.shape(xi)[0]) * P[1]

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
    f = scipy.interpolate.interp1d(
        Y[:, 0],
        Y[:, 1],
        bounds_error=False,
        fill_value=fill_value,
    )
    return f(xi)


# function[in]=find_nearest(x,y)
# % for each element of x, find the closest to y (???)

# in = zeros(size(y))*NaN;
# for n=1:length(in)
#   [~,in(n)]=min(abs(x-y(n)));
# end


# From https://stackoverflow.com/questions/40890960/numpy-scipy-equivalent-of-matlabs-sparse-function
def sparse(i, j, v, m, n):
    """
    Create and compressing a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return scipy.sparse.csr_array((v, (i, j)), shape=(m, n))


def bindata(x, y, gx):
    """Bins y(x) onto b(gx), gx defining centers of the bins. NaNs ignored.

    Args:

    Returns:
        b: binned data (averaged)
        n: number of points in each bin

    Notes:
        Current implimentation only handles the 1-D case
    """
    # x=x(:);y=y(:);
    x = np.squeeze(x)
    y = np.squeeze(y)

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


def StripVars(dsi, dso, var_meta):
    """Copies dsi to dso, excliding any varables in the var_meta dict

    Note: at this time, the dimensions are not addressed

    """
    strip_names = [var_meta[k]["nc_varname"] for k in var_meta]

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
            nc_var[:] = var[:]
            for a in var.ncattrs():
                if a != "_FillValue":
                    nc_var.setncattr(a, var.getncattr(a))

    for a in dsi.ncattrs():
        dso.setncattr(a, dsi.getncattr(a))


def CreateNCVar(dso, template, key_name, data):
    """Creates a nc variable and sets meta data
    Input:
        dso - output dataset
        template - dictionary of metadata
        var_name - name of variable as appears in teamplate
        data - input data

    Returns:
        dataarray for variable and matching qc variable

    """
    # is_str = False
    if isinstance(data, str):
        # Should be unused
        inp_data = np.array(data, dtype=np.dtype(("S", len(data))))
        # is_str = True
    elif np.ndim(data) == 0:
        # Scalar data
        inp_data = np.dtype(template[key_name].nc_type).type(data)
    else:
        inp_data = data.astype(template[key_name].nc_type)

    if hasattr(template[key_name], "decimal_pts"):
        inp_data = inp_data.round(template[key_name].decimal_pts)

    # Check for scalar variables
    if np.ndim(inp_data) == 0:
        if inp_data == np.nan:
            inp_data = template[key_name].nc_attribs.FillValue
    else:
        inp_data[np.isnan(inp_data)] = template[key_name].nc_attribs.FillValue

    # assert len(template[key_name]["nc_dimensions"]) == 1
    # if template[key_name]["nc_dimensions"][0] not in dso.dimensions:
    #    dso.createDimension(template[key_name]["nc_dimensions"][0], np.shape(data)[0])

    for ii, dim in enumerate(template[key_name].nc_dimensions):
        if dim not in dso.dimensions:
            dso.createDimension(dim, np.shape(data)[ii])

    nc_var = dso.createVariable(
        template[key_name].nc_varname,
        template[key_name].nc_type,
        template[key_name].nc_dimensions,
        fill_value=template[key_name].nc_attribs.FillValue,
        compression="zlib",
        complevel=9,
    )
    nc_var[:] = inp_data
    for a_name, attrib in iter(template[key_name].nc_attribs):
        # TODO - finish this off
        # if a_name in ("valid_min", "valid_max"):
        #    nc_var.setncattr(a_name, np.array(attrib).astype(template[key_name]["nc_type"])
        if a_name != "FillValue" and attrib is not None:
            nc_var.setncattr(a_name, attrib)


def CreateNCVars(dso, ad2cp_var_map, var_meta):
    for key_name, data in ad2cp_var_map.items():
        CreateNCVar(dso, var_meta, key_name, data)


def GetMissionStr(dive_nc_file):
    """Assembles the mission str"""
    log_id = None
    mission_title = ""
    if "log_ID" in dive_nc_file.variables:
        log_id = int(dive_nc_file.variables["log_ID"].getValue())
    if "sg_cal_mission_title" in dive_nc_file.variables:
        mission_title = dive_nc_file.variables["sg_cal_mission_title"][:].tobytes().decode("utf-8")
    return f"SG{'%03d' % (log_id if log_id else 0,)} {mission_title}"


def SetupPlotDirectory(adcp_opts) -> int:
    """Ensures plot_directory is set in base_opts and creates it if needed

    Returns:
        0 for success
        1 for failure

    """
    if adcp_opts.plot_directory.exists():
        if adcp_opts.plot_directory.is_dir():
            return 0
        else:
            log_error(f"{adcp_opts.plot_directory} is not a directory")
            return 1
    else:
        try:
            adcp_opts.plot_directory.mkdir(mode=0x777)
        except Exception:
            log_error(f"Could not create {adcp_opts.plot_directory}", "exc")
            return 1
    return 0


def isoSurface(field, target, dim):
    """
    Linearly interpolate a coordinate isosurface where a field
    equals a target

    Parameters
    ----------
    field : xarray DataArray
        The field in which to interpolate the target isosurface
    target : float
        The target isosurface value
    dim : str
        The field dimension to interpolate

    Examples
    --------
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
