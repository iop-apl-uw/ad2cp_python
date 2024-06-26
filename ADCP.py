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
ADCP.py - Main caclulation functions
"""

import pdb
from typing import Any

import numpy as np
import scipy

import ADCPConfig
import ADCPFiles
import ADCPUtils


def CleanADCP(
    adcp: ADCPFiles.ADCPRealtimeData,
    glider: ADCPFiles.SGData,
    param: ADCPConfig.Params,
) -> None:
    """Performs some housekeeping cleanup on the adcp data"""

    # ruff: noqa: SIM118

    # index_bins = param.index_bins;

    # adcp.U0 = adcp.U;
    # adcp.V0 = adcp.V;
    # adcp.W0 = adcp.W;
    # Back up originals
    for vv in ["U", "V", "W"]:
        adcp[f"{vv}0"] = np.copy(adcp[vv])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % I'm not sure I trust the pressure on the ADCP. Use the glider one instead
    # adcp.Z0 = interp1(glider.Mtime, glider.ctd_depth, adcp.Mtime);
    # adcp.Z0(isnan(adcp.Z0))=0;
    # adcp.Z = adcp.Z0 - adcp.Range.*adcp.TiltFactor;
    if param.use_glider_pressure:
        f = scipy.interpolate.interp1d(
            glider.ctd_time,
            glider.ctd_depth,
            bounds_error=False,
            # fill_value="extrapolate",
        )
        adcp.Z0 = f(adcp.time)
        adcp.Z0[np.isnan(adcp.Z0)] = 0.0
        adcp.Z = adcp.Z0 - np.atleast_2d(adcp.Range).T * adcp.TiltFactor
    else:
        # TOOO - convert adcp_pressure into depth, then generate the Z and Z0
        raise RuntimeError("NYI")

    # % fix soundspeed
    # if isfield(adcp,'Svel')
    # Svel = interp1(glider.Mtime,glider.sound_velocity,adcp.Mtime);
    # Svel = intnan(Svel);
    # adcp.U = adcp.U.*(Svel./adcp.Svel);
    # adcp.V = adcp.V.*(Svel./adcp.Svel);
    # adcp.W = adcp.W.*(Svel./adcp.Svel);
    # end

    # Fix sound speed - using the gliders estimations
    f = scipy.interpolate.interp1d(
        glider.ctd_time,
        glider.sound_velocity,
        # bounds_error=False,
        # fill_value="extrapolate",
    )
    Svel = f(adcp.time)

    ADCPUtils.intnan(Svel)

    adcp.Svel = Svel

    scale_v = Svel / adcp.Svel
    for var_n in ["U", "V", "W"]:
        adcp[var_n] *= scale_v

    # % trim range bins
    # names = fieldnames(adcp);
    # NB = size(adcp.U,1);
    # for nv = 1:length(names)
    #   if size(adcp.(names{nv}),1)==NB
    #     adcp.(names{nv})=adcp.(names{nv})(index_bins,:);
    #   end
    # end

    # Trim range bins as specified
    NB = np.shape(adcp.U)[0]
    for var_n in adcp.keys():
        var_s = np.shape(adcp[var_n])
        if len(var_s) > 1 and var_s[0] == NB:
            adcp[var_n] = adcp[var_n][param.index_bins, :]

    # GBS 2024/06/13 - Not in realtime processing
    #
    # % Correlation mask
    # if isfield(adcp,'C')
    #   if size(adcp.C,2)==size(adcp.U,2)
    #   Cmin = 20;
    #   mask = adcp.C<Cmin;
    #   adcp.U(mask) = NaN;
    #   adcp.V(mask) = NaN;
    #   end
    # end

    # Good to here

    # % surface mask
    # % mask = adcp.Z<(adcp.Z0*0.1+adcp.CellSize) | adcp.Z<sfc_blank;
    # mask = (adcp.Z<=param.sfc_blank);
    # adcp.U(mask) = NaN;
    # adcp.V(mask) = NaN;
    # adcp.W(mask) = NaN;
    surf_mask = param.sfc_blank > adcp.Z
    for var_n in ["U", "V", "W"]:
        adcp[var_n][surf_mask] = np.nan
    # MAX_TILT = 0.8;
    # mask = adcp.TiltFactor<MAX_TILT;
    # adcp.U(:,mask) = NaN;
    # adcp.V(:,mask) = NaN;
    # adcp.W(:,mask) = NaN;

    # Good to here

    # Maximum tilt
    # CONSIDER - Make a param?  Ask Luc
    MAX_TILT = 0.8
    tilt_mask = adcp.TiltFactor < MAX_TILT
    for var_n in ["U", "V", "W"]:
        adcp[var_n][:, tilt_mask] = np.nan

    # % W error?
    # WMAX_error = 0.05; % difference between vertical velocity measured by ADCP and vertical motion of the glider.
    # % aW0 = interp1(mpoint(glider.Mtime),W0,aMtime);
    # % compare the vertcal velocity to what is expected from the model...
    # aW0 = interp1(glider.Mtime,glider.Wmod,adcp.Mtime);
    # mask = abs(adcp.W + aW0)>WMAX_error;
    # adcp.U(mask) = NaN;
    # adcp.V(mask) = NaN;
    # adcp.W(mask) = NaN;

    # W error?
    # difference between vertical velocity measured by ADCP and vertical motion of the glider.
    # CONSIDER - Make a param?  Ask Luc

    # traced to here
    WMAX_error = 0.05
    f = scipy.interpolate.interp1d(
        glider.ctd_time,
        glider.Wmod,
        bounds_error=False,
        # fill_value="extrapolate",
    )
    aW0 = f(adcp.time)
    w_mask = np.abs(adcp.W + aW0) > WMAX_error
    for var_n in ["U", "V", "W"]:
        adcp[var_n][w_mask] = np.nan

    # Good to here

    # % maximum relative velocity?
    # UMAX = 0.5;
    # mask = sqrt(adcp.U.^2+adcp.V.^2)>UMAX;
    # adcp.U(mask) = NaN;
    # adcp.V(mask) = NaN;
    # adcp.W(mask) = NaN;

    # maximum relative velocity?
    UMAX = 0.5
    rel_vel_mask = np.sqrt(adcp.U**2 + adcp.V**2) > UMAX
    for var_n in ["U", "V", "W"]:
        adcp[var_n][rel_vel_mask] = np.nan

    # % delete empty profiles
    # ie = all(isnan(adcp.U));
    # names = fieldnames(adcp);
    # nens = size(adcp.U,2);
    # for nv = 1:length(names)
    #   tmp = adcp.(names{nv});
    #   if size(tmp,2)==nens
    #     adcp.(names{nv})(:,ie)=[];
    #   end
    # end

    # delete empty profiles
    ine = np.logical_not(np.all(np.isnan(adcp.U), axis=0))
    nens = np.shape(adcp.U)[1]
    for var_n in adcp.keys():
        var_s = np.shape(adcp[var_n])
        if len(var_s) == 1 and var_s[0] == nens:
            adcp[var_n] = adcp[var_n][ine]
        elif len(var_s) > 1 and var_s[1] == nens:
            adcp[var_n] = adcp[var_n][:, ine]

    return


# Matches ad2cp_inverse6 from matlab code
def Inverse(
    adcp: ADCPFiles.ADCPData | ADCPFiles.ADCPRealtimeData,
    gps: ADCPFiles.GPSData,
    glider: ADCPFiles.SGData,
    weights: ADCPConfig.Weights,
    param: ADCPConfig.Params,
) -> Any:
    gz = param.gz
    dz = param.dz

    profile = ADCPFiles.ADCPProfile()

    profile.z = gz
    profile.UVocn = np.zeros((len(gz), 2)) * np.nan
    profile.UVttw_solution = np.zeros((len(gz), 2)) * np.nan
    profile.UVttw_model = np.zeros((len(gz), 2)) * np.nan
    profile.time = np.zeros((len(gz), 2)) * np.nan
    profile.UVerr = np.zeros((len(gz), 2)) * np.nan
    profile.Wocn = np.zeros((len(gz), 2)) * np.nan
    profile.Wttw_solution = np.zeros((len(gz), 2)) * np.nan
    profile.Wttw_model = np.zeros((len(gz), 2)) * np.nan

    # Make a new regular time grid, from the first to the last gps fixes, which
    # has the same time interval as most ADCP emsembles.

    # typical ADCP interval
    tmp = np.diff(adcp.time)
    #  largest of 15 seconds or ADCP intervals
    dt = max(15, np.median(tmp[tmp > 2]))

    D = ADCPFiles.ADCPTemp()

    # D.time = param.time_limits[0]:dt/86400:param.time_limits(end);
    # D.Z0 = interp1(glider.Mtime, glider.ctd_depth, D.Mtime);
    D.time = np.arange(param.time_limits[0], param.time_limits[-1], dt)

    # Note - this is the first place we get differences in code - it comes down to the matlab handling
    # of interp1d for out-of-bounds values (tails of D.time are outside glider.ctd_time).  Pushing ahead
    # to see of its an issue.
    D.Z0 = scipy.interpolate.interp1d(
        glider.ctd_time,
        glider.ctd_depth,
        bounds_error=False,
        # fill_value="extrapolate",
        fill_value=np.nan,
    )(D.time)

    # % align (assign each ADCP ensemble to the nearest regular time grid).
    # in = find_nearest(D.Mtime,adcp.Mtime);
    # toff = adcp.Mtime-D.Mtime(in);

    # align (assign each ADCP ensemble to the nearest regular time grid).
    idx = np.zeros(adcp.time.shape[0], np.int32)
    for ii in range(idx.shape[0]):
        idx[ii] = np.argmin(np.abs(D.time - adcp.time[ii]))
    toff = adcp.time - D.time[idx]
    # D.Mtime = D.Mtime+mean(toff); % if the regular time is constantly off by a few seconds, shift the regular time.

    # if the regular time is constantly off by a few seconds, shift the regular time.
    D.time = D.time + np.mean(toff)

    # D.Mtime(in) = adcp.Mtime; % now some of the times match, and others are close
    # D.UV = nan(size(adcp.Z,1),length(D.Mtime)); D.UV(:,in) = (adcp.U+1i*adcp.V);  % observed horizontal velocity (relative)
    # D.W = nan(size(adcp.Z,1),length(D.Mtime)); D.W(:,in) =  adcp.W;               % observed vertical velocity (relative)
    # D.Z = nan(size(adcp.Z,1),length(D.Mtime));D.Z(:,in) = adcp.Z;

    # now some of the times match, and others are close
    D.time[idx] = adcp.time
    # observed horizontal velocity (relative)
    D.UV = np.full((np.shape(adcp.Z)[0], np.shape(D.time)[0]), np.nan * 1j * np.nan)
    D.UV[:, idx] = adcp.U + 1j * adcp.V
    # observed vertical velocity (relative)
    D.W = np.full((np.shape(adcp.Z)[0], np.shape(D.time)[0]), np.nan)
    D.W[:, idx] = adcp.W
    D.Z = np.full((np.shape(adcp.Z)[0], np.shape(D.time)[0]), np.nan)
    D.Z[:, idx] = adcp.Z

    # D.Z0(in) = adcp.Z0;
    # ii = find(isfinite(D.Z0));
    # D.Z0 = interp1(D.Mtime(ii),D.Z0(ii),D.Mtime);    % depth of the ADCP (glider).
    # % peg the surface intervals
    # D.Z0(D.Z0<1 | isnan(D.Z0)) = 0;
    # % separate down/up casts
    # [~,ibot] = max(D.Z0);
    # D.upcast = (1:length(D.Mtime))>ibot;

    # Earlier differences in D.Z0 interp  are eliminated by this overwriting of problem locations
    D.Z0[idx] = adcp.Z0
    ii = np.nonzero(np.isfinite(D.Z0))[0]
    D.Z0 = scipy.interpolate.interp1d(
        D.time[ii],
        D.Z0[ii],
        bounds_error=False,
        # fill_value="extrapolate",
        fill_value=np.nan,
    )(D.time)

    # peg the surface intervals
    D.Z0[np.logical_or(D.Z0 < 1, np.isnan(D.Z0))] = 0
    # separate down/up casts
    ibot = np.argmax(D.Z0)
    D.upcast = np.arange(D.time.shape[0]) > ibot

    return (D, profile, None)
