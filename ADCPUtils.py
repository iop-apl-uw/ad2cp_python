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
import sys

import netCDF4
import numpy as np
import numpy.typing as npt
import scipy

from ADCPLog import log_error


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

    return 0


def open_netcdf_file(ncf_name: pathlib.Path) -> None | netCDF4.Dataset:
    try:
        ds = netCDF4.Dataset(ncf_name, "r")
    except Exception:
        log_error(f"Failed to open {ncf_name}", "exc")
        return None
    ds.set_auto_mask(False)
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


def latlon2xy(ll: npt.NDArray[np.complex128], ll0: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    ll -= ll0

    # xy =      lon_to_m(real(ll),imag(ll0))+...
    #         i*lat_to_m(imag(ll),imag(ll0));

    xy = lon_to_m(ll.real, ll0.imag) + 1j * lat_to_m(ll.imag, ll0.imag)
    return xy * 1e-3


# function[yi] = IT_sg_interp_AP(x,y,xi)
# % complex interpolation, where the amplitude and phase of y are linearly interpolated...
# % This does not require a regular xi.
# %
# % Luc Rainville 4/13/12

# ii = find(isfinite(x));

# % Amplitude - easy
# if length(x(ii))==1
#   Ai = abs(y(ii));
# else
#   Ai = interp1(x(ii),abs(y(ii)), xi);
# end

# % Phase - a little more tricky
# P = unwrap(angle(y(ii)));
# if length(x(ii))==1
#   Pi = P(ii);
# else
#   Pi = interp1(x(ii),P, xi);
# end
# yi = Ai.*exp(1i*Pi);


def IT_sg_interp_AP(x, y, xi):
    return None


# function[yi]=course_interp(x,y,xi)
# % Y is a heading in degrees.
# y0 = IT_sg_interp_AP(x,exp(1i*y*pi/180),xi);
# yi = angle(y0)*180/pi;
# yi(yi<0)=yi(yi<0)+360;


def course_interp(time, heading, ctd_time):
    return None


#
# black_filt
#

# function [yf, ww]  = bl_filt(y, half_width,varargin)
# % function [yf, ww]= black_filt(y, half_width)
# %
# % Blackman filter y.  The filter slides over the whole array,
# % using only the non-NaN values but yielding results for all
# % elements, so long as the number of consecutive points does not exceed
# % twice the half-width.
# %
# % ww is the filter fraction with data; it is 1 if there are no gaps within
# % the filter width, 0.5 if half the filter integral covered gaps,
# % NaN (like yf) if there were no good points within the filter range.
# %
# % Note: the interpolation becomes shaky for small values of the filter
# % fraction; either NaNs or some other method of interpolation should
# % be used for filter fractions less than something like 0.3-0.5.
# %
# % black_filt(...,'NOPADDING') turns off the default cosine-tapered padding
# % at the boundary (equivalent to padding with NaNs or with even reflection)
# %
# % black_filt(...,'NODETREND') turns off the default de-trending
# %
# % Default options (padding + de-trending) tend to preserve endpoints


# % E.F. 94/12/10  (R/V Knorr, WHP I8S)
# %
# % A.S. 00/12/30 modified to handle matricies (works for each column);
# % A.S. (ca. 2006): function is reflected symmetrically about the endpoints to create appropriate padding.
# %                  this ensured that the end point never changes (bug or feature?)
# % A.S. (ca. 2006): De-trending added by default
# % A.S. (ca. 2007): the reflected parts are cosine-tapered (idea from PLT64 filter).

# % Default cos-tapered padding is somewhere in-between odd and even reflection:
# %
# % EVEN (no padding)       ODD padding             ODD with cosine taper
# %                          (default)
# %
# %                                 /
# %                                /                        .-.
# %      |                      | /                      | /   \
# %      |                      |/                       |/     \
# %     /|\                    /|                       /|
# %    / | \                  / |                      / |
# %   /     \                /                        /
# %  /       \              /                        /

# mn=get_option(varargin,'minpoints',1);
# do_padding = ~is_option(varargin,'NOPADDING');
# do_detrending = ~is_option(varargin,'NODETREND');

# COSINE = is_option(varargin,'COSINE'); % use cos damping of edge reflections?

# %if (nargin ~= 2)
# %  help bl_filt
# %  error('wrong number of parameters');
# %end

# if half_width<=1,
#     yf=y;
#     ww=ones(size(y));
#     return;
# end

# sz0 = size(y);
# if sz0(1)==1,
#     y=y(:);
# end;
# sz = size(y);
# y = reshape(y,sz(1),prod(sz(2:end)));


# %--- generate weights array for given width ---
# half_width = round(half_width);
# nf = half_width * 2 + 1;
# [ny, nn] = size(y);


# x = (-half_width:half_width)*pi/half_width;
# w = 0.42 + 0.5 * cos(x) + 0.08 * cos(2*x);
# w=w.';

# if (ny<nf),
# %     warning('BLACK_FILT: time series is shorter than the filter');
#     d = floor((nf-ny)/2)+1; %extra points to discard on each end
#     w([1:d,end-d+1:end]) = [];
#     half_width = (length(w)-1)/2;
# end

# if half_width<1,
#     % just return the average!
#     yf = repmat(nanmean(y),ny,1);
#     yf = reshape(yf,sz0);
#     return
# end


# if do_detrending,
#     % detrend (taking nans into account)
#     P = [(1:ny)', ones(ny,1)]; % regressor matrix
# %     [Q,R] = qr(P,0);
#     for k=1:nn
#         idok = find(isfinite(y(:,k)));
#         ab(:,k) = P(idok,:)\y(idok,k); % fit coefficients
# %         ab(:,k) = R\(Q(idok,:)'*y(idok,k)); % wrong??

#         y(:,k) = y(:,k) - P*ab(:,k); %detrended series
#     end
# end


# if do_padding
#     n = half_width; %(actual padding length is n-1)
#     if COSINE,
#         %pad the time series with cosine-damped reflection...
#         y1 = repmat(y(1,:),n-1,1)+...
#             (repmat(y(1,:),n-1,1)-y(n:-1:2,:)).*...
#             repmat(cos(((n-1):-1:1)/(n-1)*pi/2)',1,nn);% top padding
#         y2 = repmat(y(end,:),n-1,1)+...
#             (repmat(y(end,:),n-1,1)-y(end-1:-1:end-n+1,:)).*...
#             repmat(cos((1:(n-1))/(n-1)*pi/2)',1,nn);% bottom padding
#     else
#         %pad the time series with undamped reflection...
#         y1 = 2*repmat(y(1,:),n-1,1)-y(n:-1:2,:);
#         y2 = 2*repmat(y(end,:),n-1,1)-y(end-1:-1:end-n+1,:);
#     end
#     y = [y1;y;y2]; % padded time series

# else
#     n = 1;
# end


# bad_mask = ~isfinite(y);
# good_mask = ~bad_mask;

# % Replace NaNs with zeros.
# ib = find(bad_mask);
# if ~isempty(ib),
#     y(ib) = good_mask(ib);
# end


# ytop = conv2(y, w,'same');
# ybot = conv2(double(good_mask), w,'same');

# % If a gap
# ibad = find(ybot < mn);
# if (~isempty(ibad))
#     ybot(ibad) = NaN * ybot(ibad);
# end

# yf = ytop./ ybot;
# % remove padding
# yf = yf(n-1+(1:ny),:);

# if do_detrending,
#     % add back the trend
#     yf = yf + P*ab;
# end

# yf = reshape(yf,sz0);

# if nargout > 1,
#     % reflected ends can't be considered "good data"
#     good_mask(1:n-1,:) = 0;
#     good_mask(n+ny:end,:) = 0;

#     ndata = conv2(double(good_mask), abs(w),'same');
#     ww = ndata / sum(abs(w));
#     % throw away the padding:
#     ww = ww(n-1+(1:ny),:);
#     ww = reshape(ww,[ny,nn]);

# end

# function [r,new_args]=get_option(args,name,default,typecast)
# % GET_OPTION extract "option-value" pairs from VARARGIN list
# %	 r=get_option(args,name,default)
# %	 if the option 'name' present in args{:}, returns the next string in args{:}
# %   if 'default' is not specified, [] assumed.
# % 	 [r,new_args]=get_option(...) returns the option list without the option pair processed
# %	 Note: usually called with varargin

# %
# % 2001-2012, ashcherbina@apl.uw.edu


# n=length(args);
# k=find(strcmpi(args,name),1,'last');
# if (isempty(k) || k>(n-1))
#    if nargin>2
#       r=default;
#    else
#       r=[];
#    end
# else
#    io=max(k);
#    r=args{io+1};
#    if nargin>3 && strcmp(typecast,'num') && ischar(r),
#        r = str2num(r);
#    end
#    % display the option
# %    fprintf('%s: ',args{io});   disp(r);
#    args=args([1:io-1,io+2:end]);
# end

# if nargout>1
#    new_args=args;
# end


# function [r,args]=is_option(args,name)
# % IS_OPTION looks for an option in the option list
# % 	r=is_option(args,name)
# % 	returns logical(1) if the option 'name' is present in args{:}, logical(0) otherwise;
# %  Can also be called is_option(args,{name1,name2,...}) to check for multiple options ("OR" assumed)
# %  [r,new_arg]=is_option(...) also returns the argument list without the options found.
# % 	Note: usually called with varargin

# %
# % 2001-2012, ashcherbina@apl.uw.edu
# if iscell(name)
#    r=0;
#    for on=1:length(name)
#       k=find(strcmpi(args,name{on}));
#       r=~isempty(k);
#       if (r), break;end
#    end
# else
#    k=find(strcmpi(args,name));
#    r=~isempty(k);
# end
# if nargout>1 && (r),
#       if k<length(args),
#          args = args([1:k-1,k+1:end]);
#       else
#          args = args(1:k-1);
#       end
# end
