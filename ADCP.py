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

# TODO - review all code blocks and pull out the tight intermixed matlab

# import pdb
import warnings
from typing import Any

import numpy as np
import scipy as sp

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
        f = sp.interpolate.interp1d(
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
    f = sp.interpolate.interp1d(
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
    f = sp.interpolate.interp1d(
        glider.ctd_time,
        glider.Wmod,
        bounds_error=False,
        # fill_value="extrapolate",
    )
    aW0 = f(adcp.time)
    w_mask = np.abs(adcp.W + aW0) > WMAX_error
    for var_n in ["U", "V", "W"]:
        adcp[var_n][w_mask] = np.nan

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
    inverse_tmp: dict | None = None,
) -> Any:
    # gz = param.gz
    # dz = param.dz

    # DEBUG np.set_printoptions(precision=4, linewidth=200)

    profile = ADCPFiles.ADCPProfile()

    profile.z = param.gz
    profile.UVocn = np.zeros((len(param.gz), 2), dtype=np.complex128) * np.nan
    profile.UVttw_solution = np.zeros((len(param.gz), 2), dtype=np.complex128) * np.nan
    profile.UVttw_model = np.zeros((len(param.gz), 2), dtype=np.complex128) * np.nan
    profile.time = np.zeros((len(param.gz), 2)) * np.nan
    profile.UVerr = np.zeros((len(param.gz), 2)) * np.nan
    profile.Wocn = np.zeros((len(param.gz), 2)) * np.nan
    profile.Wttw_solution = np.zeros((len(param.gz), 2)) * np.nan
    profile.Wttw_model = np.zeros((len(param.gz), 2)) * np.nan

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

    # TODO Review with Luc
    # Note - this is the first place we get differences in code - it comes down to the matlab handling
    # of interp1d for out-of-bounds values (tails of D.time are outside glider.ctd_time).  Pushing ahead
    # to see of its an issue.
    D.Z0 = sp.interpolate.interp1d(
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
    # % observed horizontal velocity (relative)
    # D.UV = nan(size(adcp.Z,1),length(D.Mtime)); D.UV(:,in) = (adcp.U+1i*adcp.V);
    # % observed vertical velocity (relative)
    # D.W = nan(size(adcp.Z,1),length(D.Mtime)); D.W(:,in) =  adcp.W;
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
    D.Z0 = sp.interpolate.interp1d(
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

    # % model velocity
    # % UV1 does not include DAC
    # D.UVttw_model = interp1(glider.Mtime, glider.UV1, D.Mtime); % make sure the model is available at all times
    # D.UVttw_model(isnan(D.UVttw_model)) = 0;

    # D.Wttw_model = interp1(glider.Mtime, glider.Wmod, D.Mtime); % make sure the model is available at all times

    # model velocity
    # UV1 does not include DAC
    # make sure the model is available at all times
    D.UVttw_model = sp.interpolate.interp1d(
        glider.ctd_time,
        glider.UV1,
        bounds_error=False,
        # fill_value="extrapolate",
        fill_value=np.nan,
    )(D.time)
    D.UVttw_model[np.isnan(D.UVttw_model)] = 0

    # make sure the model is available at all times
    D.Wttw_model = sp.interpolate.interp1d(
        glider.ctd_time,
        glider.Wmod,
        bounds_error=False,
        # fill_value="extrapolate",
        fill_value=np.nan,
    )(D.time)

    ## prepare the inverse
    # the unknown state vector consists of U_ttw (one for each ping) and
    # U_ocean (on a regular vertical grid gz), down- and up-casts back-to-front
    # Vehicle velocity is TTW, Ocean velocity is Earth-referenced,
    # So,
    # ADCP measurement = U_ocean - (U_ttw+U_drift);
    # Total vehicle motion = (U_ttw+U_drift)
    # through-the-water motion =  U_ttw;
    # where U_drift is U_ocean interpolated onto the vehicle location

    # gz=gz(gz<max(glider.ctd_depth)+median(diff(gz)));
    # Nz = numel(gz);
    gz = param.gz[param.gz < np.max(glider.ctd_depth) + np.nanmedian(np.diff(param.gz))]
    Nz = gz.size
    (Nbin, Nt) = np.shape(D.UV)

    # dc = D.UV;

    # measured speed relative to the glider.
    dc = D.UV

    # ia = ~isnan(dc);
    # d_adcp = W_MEAS*dc(ia);

    # select valid data
    ia = np.logical_not(np.isnan(dc))
    # RHS, long vector of weighted measured velocities
    # *matlab* - transpose to preserve matlab order
    # TODO - Check with Luc here - d_adcp goes to Mx1 that is dependent on the
    # FORTRAN order of unrolling
    d_adcp = weights.W_MEAS * dc.T[ia.T]
    Na = d_adcp.size
    if inverse_tmp is not None:
        inverse_tmp["d_adcp"] = d_adcp

    # dw_adcp = W_MEAS*D.W(ia);

    # RHS, long vector of weighted measured vertical velocities
    # *matlab* - transpose to preserve matlab order
    dw_adcp = weights.W_MEAS * D.W.T[ia.T]

    if inverse_tmp is not None:
        inverse_tmp["dw_adcp"] = dw_adcp
    # z = D.Z(ia); % ...and depth
    # D.T=ones(size(dc,1),1)*D.Mtime; % Time of each observations
    # TT=D.T(ia);
    # Z0 = ones(size(D.Z,1),1)*D.Z0; % depth of the glider at each observation
    # Z0 = Z0(ia);

    # *matlab* - transpose to preserve matlab order
    z = D.Z.T[ia.T]  # ...and depth
    # Time of each observations
    # *matlab* - transpose to preserve matlab order
    TT = (np.atleast_2d(np.ones(np.shape(dc)[0])).T * D.time).T[ia.T]
    # depth of the glider at each observation
    # *matlab* - transpose to preserve matlab order
    Z0 = (np.atleast_2d(np.ones(np.shape(D.Z)[0])).T * D.Z0).T[ia.T]

    if inverse_tmp is not None:
        inverse_tmp["z"] = z
        inverse_tmp["TT"] = TT
        inverse_tmp["Z0"] = Z0

    # pdb.set_trace()

    # upcast = repmat(D.upcast,size(D.Z,1),1);
    # upcast = upcast(ia);
    # Na = numel(d_adcp);

    upcast = np.tile(D.upcast, (np.shape(D.Z)[0], 1)).T[ia.T]
    if inverse_tmp is not None:
        inverse_tmp["upcast"] = upcast

    # %%%%%%%%%%
    # % Av is matrix selecting the vehicle ttw velocity at the time of each ADCP measurement
    # jprof = cumsum(ones(Nbin,Nt),2);  %corresponding profile number (i.e. U_ttw index)
    # jprof = jprof (ia);
    # Av = sparse(1:Na, jprof, ones(1,Na),Na,Nt);

    # corresponding profile number (i.e. U_ttw index)
    jprof = np.cumsum(np.ones((Nbin, Nt)), 1)
    jprof = jprof.T[ia.T]

    # *matlab* - jprof - 1 to convert from matlab to python indexing
    Av = ADCPUtils.sparse(np.arange(Na), jprof - 1, np.ones(Na), Na, Nt)
    if inverse_tmp is not None:
        inverse_tmp["jprof"] = jprof
        inverse_tmp["Av"] = Av.todense()

    # %%%%%%%%%%
    # % AiM is a matrix assigning the vertical (ocean profile) grid to the measurement position

    # % Since the bins don't exactly match the ocean grid, we use 'fractional'
    # % indices to indicate how much each ocean bin contributes to the
    # % measurements. In the end, sum(AiM,2)==1 everywhere.

    # % % interpolation from the vertical grid to the measurement positions
    # rz = (z(:)-gz(1))/param.dz; % fractional z-index
    # iz = [floor(rz)+1,floor(rz)+2]; % interpolant indices
    # wz = [1-(rz-floor(rz)), rz-floor(rz)]; % interpolant weights
    # % to allow for down-/up-cast...
    # iz(upcast,:) = iz(upcast,:)+Nz;
    # % Two ocean profiles
    # AiM = sparse(repmat( (1:Na)',1,2 ), iz, wz,Na,2*Nz);

    # interpolation from the vertical grid to the measurement positions
    rz = (z - gz[0]) / param.dz  # fractional z-index
    iz = np.array((np.floor(rz) + 1, np.floor(rz) + 2)).T  # interpolant indices - one (matlab) based
    # to allow for down-/up-cast...
    iz[upcast, :] = iz[upcast, :] + Nz
    wz = np.array((1 - (rz - np.floor(rz)), rz - np.floor(rz))).T  # interpolant weights

    if inverse_tmp is not None:
        inverse_tmp["rz"] = rz
        inverse_tmp["iz"] = iz
        inverse_tmp["wz"] = wz

    # Two ocean profiles
    # *matlab* - reshape needed because python sparse only allows 1d for initialization
    # *matlab* - iz is indexes that are one based
    AiM = ADCPUtils.sparse(
        np.tile(np.atleast_2d(np.arange(Na)).T, 2).reshape(2 * Na),
        iz.reshape(2 * Na) - 1,
        wz.reshape(2 * Na),
        Na,
        2 * Nz,
    )
    if inverse_tmp is not None:
        inverse_tmp["AiM"] = AiM.todense()

    # AiO is a matrix assigning the vertical (ocean profile) grid at the vehicle position
    # AiG is a matrix assigning the vertical (ocean profile) grid at the vehicle position,
    #     expressed at every measurement position

    # interpolation from the vertical grid to the vehicle position
    # rz = (D.Z0(:)-gz(1))/param.dz; % fractional z-index
    # iz = [floor(rz)+1,floor(rz)+2]; % interpolant indices ** this shouldn't be larger than Nz,
    # but technically can!?!
    # iz(D.upcast,:) = iz(D.upcast,:)+Nz;
    # wz = [1-(rz-floor(rz)), rz-floor(rz)]; % interpolant weights

    # interpolation from the vertical grid to the vehicle position
    rrz = (D.Z0 - gz[0]) / param.dz  # fractional z-index
    # interpolant indices ** this shouldn't be larger than Nz,
    # but technically can!?!
    iiz = np.array((np.floor(rrz) + 1, np.floor(rrz) + 2)).T
    # to allow for down-/up-cast...
    iiz[D.upcast, :] = iiz[D.upcast, :] + Nz
    wwz = np.array((1 - (rrz - np.floor(rrz)), rrz - np.floor(rrz))).T  # interpolant weights

    if inverse_tmp is not None:
        inverse_tmp["rrz"] = rrz
        inverse_tmp["iiz"] = iiz
        inverse_tmp["wwz"] = wwz

    # Two ocean profiles, each with Nz depths
    # Ai0 = sparse(repmat( (1:Nt)',1,2 ), iz, wz,Nt, 2*Nz);
    # AiG = Av * Ai0
    Ai0 = ADCPUtils.sparse(
        np.tile(np.atleast_2d(np.arange(Nt)).T, 2).reshape(2 * Nt),
        iiz.reshape(2 * Nt) - 1,
        wwz.reshape(2 * Nt),
        Nt,
        2 * Nz,
    )
    AiG = Av * Ai0
    if inverse_tmp is not None:
        inverse_tmp["Ai0"] = Ai0.todense()
        inverse_tmp["AiG"] = AiG.todense()

    # %%%%%%%%%%

    # % G_adcp = W_MEAS*[-Av, AiM];  % "G" matrix (Eq. B4; in Todd et al. 2011)
    # G_adcp = W_MEAS*[-Av, AiM-Av*Ai0];  %
    G_adcp = weights.W_MEAS * sp.sparse.hstack([-Av, AiM - Av * Ai0])
    if inverse_tmp is not None:
        inverse_tmp["G_adcp"] = G_adcp.todense()

    # %% CONSTRAINTS

    # SURFACE and GPS
    # Note that we may have multiple GPS fixes - add a constraint per each
    # pair. This may automatically constrain the surface velocity!

    # Constrain the glider speed to be zero before and after the dive.

    # if W_SURFACE~=0
    #   ii_surface = find(D.Z0<=param.sfc_blank);
    #   G_sfc = sparse(length(ii_surface),Nt+2*Nz);
    #   for kk=1:length(ii_surface)
    #     G_sfc(kk,ii_surface(kk))=1;
    #   end
    #   d_sfc = zeros(length(ii_surface),1);
    # else
    #   G_sfc =[];
    #   d_sfc =[];
    # end

    if weights.W_SURFACE:
        ii_surface = np.nonzero(param.sfc_blank >= D.Z0)[0]
        G_sfc = sp.sparse.lil_array((ii_surface.shape[0], Nt + 2 * Nz))
        for kk in range(ii_surface.shape[0]):
            G_sfc[kk, ii_surface[kk]] = 1
        G_sfc.tocsr()
        d_sfc = np.zeros(ii_surface.shape[0])
    else:
        G_sfc = []
        d_sfc = []
    if inverse_tmp is not None:
        inverse_tmp["G_sfc"] = G_sfc.todense()

    # clear gps_constraints
    # for k=1:length(gps.Mtime)-1 % all GPS
    #     %  for k=length(gps.Mtime)-1% just two last GPS
    #     Dt = diff(gps.Mtime([k k+1]))*86400; % sec, dive time
    #     UVbt = diff(gps.XY([k k+1]))./Dt; % mean velocity during this segment (vehicle + ocean)
    #     ii = find(within(D.Mtime,gps.Mtime([k k+1])));
    #     if length(ii)<5
    #       continue
    #     end
    #     if Dt>3600 || max(D.Z0(ii))==max(D.Z0)
    #       % dive DAC

    #       gps_constraints.dac(k) = 1;
    #       gps_constraints.TL(k,1:2) = D.Mtime(ii([1 end]));
    #       gps_constraints.UVbt(k) = UVbt;

    #       dti = diff(D.Mtime(ii))*86400;
    #       % make it so that the individual weights *average* (or sum?) to 1
    #       % (needs to be consistent below!)
    #       %     dt = dt/mean(dt)*WBT ; %AVG! This is a stronger constatint (I think...)
    #       dti = dti/sum(dti) ; %SUM!

    #       % time-average using trapezoid rule
    #       w = D.Mtime*0;
    #       w(ii(1:end-1)) = 0.5*dti;
    #       w(ii(2:end)) = w(ii(2:end)) + 0.5*dti; % "w" is non zero only during the time of these measurements...

    #       G_dac = [+w w*Ai0]*W_DAC;
    #       d_dac = UVbt*W_DAC;

    #     elseif Dt<3600 && W_SURFACE~=0
    #       % Surface drift (-> Constrain OCEAN VELOCITY to be like drift, glider speed to be zero)

    #       gps_constraints.dac(k) = 0;
    #       gps_constraints.UVbt(k) = UVbt;
    #       gps_constraints.TL(k,1:2) = D.Mtime(ii([1 end]));

    #       dti = diff(D.Mtime(ii))*86400;
    #       dti = dti/sum(dti) ; %SUM!
    #       % time-average using trapezoid rule
    #       w = D.Mtime*0;
    #       w(ii(1:end-1)) = 0.5*dti;
    #       w(ii(2:end)) = w(ii(2:end)) + 0.5*dti; % "w" is non zero only during the time of these measurements...

    #       G_sfc = [G_sfc ; [zeros(size(w)) w*Ai0]*W_SURFACE];
    #       d_sfc = [d_sfc ; UVbt*W_SURFACE ];
    #     end
    # end

    # TODO - change so this is array, not dictionary based,  so comparisions work
    gps_constraints = ADCPFiles.GPSConstraints()

    for k in range(gps.log_gps_time.shape[0] - 1):
        Dt = np.diff(gps.log_gps_time[np.r_[k, k + 1]])
        UVbt = np.diff(gps.XY[np.r_[k, k + 1]]) / Dt  # mean velocity during this segment (vehicle + ocean)
        ii = np.nonzero(np.logical_and(D.time >= gps.log_gps_time[k], D.time <= gps.log_gps_time[k + 1]))[0]
        if ii.shape[0] < 5:
            continue

        if Dt > 3600 or max(D.Z0[ii]) == max(D.Z0):
            # dive DAC

            gps_constraints.dac[k] = 1
            gps_constraints.TL[k] = D.time[ii[[np.r_[0, -1]]]]
            gps_constraints.UVbt[k] = UVbt

            dti = np.diff(D.time[ii])
            # make it so that the individual weights *average* (or sum?) to 1
            # (needs to be consistent below!)
            #     dt = dt/mean(dt)*WBT ; %AVG! This is a stronger constatint (I think...)
            dti = dti / np.sum(dti)  # SUM!

            # time-average using trapezoid rule
            w = D.time * 0
            w[ii[:-1]] = 0.5 * dti
            w[ii[1:]] = w[ii[1:]] + 0.5 * dti  # "w" is non zero only during the time of these measurements...

            # TODO - what does the leading + do?
            # G_dac = [+w w*Ai0]*W_DAC;
            # d_dac = UVbt*W_DAC;
            G_dac = sp.sparse.csr_array(np.atleast_2d(np.hstack([+w, w * Ai0]) * weights.W_DAC))
            d_dac = sp.sparse.csr_array(np.atleast_2d(UVbt * weights.W_DAC))
            if inverse_tmp is not None:
                inverse_tmp["G_dac"] = G_dac.todense()
                inverse_tmp["d_dac"] = d_dac.todense()

        elif Dt < 3600 and weights.W_SURFACE:
            # Surface drift (-> Constrain OCEAN VELOCITY to be like drift, glider speed to be zero)

            gps_constraints.dac[k] = 0
            gps_constraints.TL[k] = D.time[ii[[np.r_[0, -1]]]]
            gps_constraints.UVbt[k] = UVbt

            dti = np.diff(D.time[ii])
            dti = dti / np.sum(dti)  # SUM!
            # time-average using trapezoid rule
            w = D.time * 0
            w[ii[:-1]] = 0.5 * dti
            w[ii[1:]] = w[ii[1:]] + 0.5 * dti  # "w" is non zero only during the time of these measurements...

            # G_sfc = [G_sfc ; [zeros(size(w)) w*Ai0]*W_SURFACE];
            # d_sfc = [d_sfc ; UVbt*W_SURFACE ];
            G_sfc = sp.sparse.vstack(
                [
                    G_sfc,
                    sp.sparse.csr_array(np.atleast_2d(np.hstack((np.zeros(w.shape[0]), w * Ai0)) * weights.W_SURFACE)),
                ]
            )
            d_sfc = np.hstack((d_sfc, UVbt * weights.W_SURFACE))
            if inverse_tmp is not None:
                inverse_tmp["G_sfc"] = G_sfc.todense()
                inverse_tmp["d_sfc"] = d_sfc

    # %% %%%%%%  % Flight model dac
    # if exist('W_MODEL_DAC','var')

    #   ii = find(D.Z0)>3;
    #   % time-average using trapezoid rule
    #   dti = diff(D.Mtime(ii))*86400;
    #   dti = dti/sum(dti) ; %SUM!
    #   w = D.Mtime*0;
    #   w(ii(1:end-1)) = 0.5*dti;
    #   w(ii(2:end)) = w(ii(2:end)) + 0.5*dti; % "w" is non zero only during the time of these measurements...

    #   MODEL_DAC = glider.depth_avg_curr_east+1i*glider.depth_avg_curr_north;

    #   G_dac = [G_dac ; [+w*0 w*Ai0]*W_MODEL_DAC]; % averaged ocean velocity is the flight model dac
    #   d_dac = [d_dac ; MODEL_DAC*W_MODEL_DAC];
    # end

    if weights.W_MODEL_DAC:
        # TODO - check with Luc - looks like matlab code is off
        # ii = find(D.Z0)>3;
        ii = np.nonzero(D.Z0 > 3)[0]
        # time-average using trapezoid rule
        dti = np.diff(D.time[ii])
        dti = dti / np.sum(dti)  # SUM!
        w = D.time * 0
        w[ii[:-1]] = 0.5 * dti
        w[ii[1:]] = w[ii[1:]] + 0.5 * dti  # "w" is non zero only during the time of these mes...

        MODEL_DAC = glider.depth_avg_curr_east + 1j * glider.depth_avg_curr_north

        # G_dac = [G_dac ; [+w*0 w*Ai0]*W_MODEL_DAC];
        # d_dac = [d_dac ; MODEL_DAC*W_MODEL_DAC];

        # averaged ocean velocity is the flight model dac
        G_dac = sp.sparse.vstack(
            [G_dac, sp.sparse.csr_array(np.atleast_2d(np.hstack([+w * 0, w * Ai0]) * weights.W_MODEL_DAC))]
        )
        d_dac = sp.sparse.vstack([d_dac, np.atleast_2d(MODEL_DAC * weights.W_MODEL_DAC)])
        if inverse_tmp is not None:
            inverse_tmp["G_dac"] = G_dac.todense()
            inverse_tmp["d_dac"] = d_dac.todense()

    # %% %% REGULARIZATION
    # % regularization of ocean velocity profile
    # % dd = spdiags(repmat([-1 1]/param.dz, 2*Nz-1,1),[0 1], 2*Nz-1,2*Nz); % d/param.dz
    # % % instead of Nz:Nz+1 continuity at the bottom of the profile (Nz+1 is surface!), enforce Nz,2*Nz continuity:
    # % dd(Nz,:) = 0;
    # % dd(Nz,Nz) = -0.5; dd(Nz,2*Nz) = +0.5;
    # % Do = [ sparse(2*Nz-1,Nt), dd ]*OCN_SMOOTH;
    # % clear dd

    # if OCN_SMOOTH~=0
    #   % Smoothness of ocean velocity
    #   dd = spdiags(repmat([-1 2 -1]/param.dz, 2*Nz-2,1),[0 1 2], 2*Nz-2,2*Nz);
    #   dd(Nz,:) = 0;
    #   dd(Nz,Nz-1) = -1/param.dz; dd(Nz,Nz) = 2/param.dz; dd(Nz,2*Nz) = -1/param.dz;
    #   Do = [ sparse(2*Nz-2,Nt), dd ]*OCN_SMOOTH;
    # else
    #   Do = [];
    # end

    if weights.OCN_SMOOTH:
        #   Smoothness of ocean velocity
        #   dd = spdiags(repmat([-1 2 -1]/param.dz, 2*Nz-2,1),[0 1 2], 2*Nz-2,2*Nz);
        #   dd(Nz,:) = 0;
        diags = np.tile(np.array([-1, 2, -1]) / param.dz, (2 * Nz - 2, 1))
        # Note - don't use this (the matrix format) - it doesn't handle the ends of the matrix correctly
        # dd = sp.sparse.spdiags(diags.T, [0, 1, 2], 2 * Nz - 2, 2 * Nz).tolil()
        dd = sp.sparse.diags_array(diags.T, offsets=[0, 1, 2], shape=(2 * Nz - 2, 2 * Nz)).tolil()
        dd[Nz - 1, :] = 0
        # TODO - Check with Luc on this - what's going on here?
        #   dd(Nz,Nz-1) = -1/param.dz;
        #   dd(Nz,Nz) = 2/param.dz;
        #   dd(Nz,2*Nz) = -1/param.dz;
        dd[Nz - 1, Nz - 2] = -1 / param.dz
        dd[Nz - 1, Nz - 1] = 2 / param.dz
        dd[Nz - 1, (2 * Nz) - 1] = -1 / param.dz
        #   Do = [ sparse(2*Nz-2,Nt), dd ]*OCN_SMOOTH;
        dd = dd.tocsr()
        Do = sp.sparse.hstack([sp.sparse.csr_array((2 * Nz - 2, Nt)), dd]) * weights.OCN_SMOOTH
        if inverse_tmp is not None:
            inverse_tmp["dd"] = dd.todense()
    else:
        Do = []
    if inverse_tmp is not None:
        inverse_tmp["Do"] = Do.todense()

    # % Up and down ocean profile should be similar... (weighted by time interval)
    # if exist('W_OCN_DNUP','var')
    #   dd = speye(Nz,Nz);
    #   [~,imax]=max(glider.ctd_depth);
    #   time1 = interp_nm(glider.ctd_depth(1:imax), glider.Mtime(1:imax), gz);
    #   time2 = interp_nm(glider.ctd_depth(imax:end), glider.Mtime(imax:end), gz);
    #   ss = 1-(time2-time1); % 1-diff in days
    #   ss(ss<0)=0; % limit to zero.
    #   for k=1:length(ss)
    #     dd(k,k)=ss(k);
    #   end
    #   ii = find(ss>0 & isfinite(ss));
    #   dd=dd(ii,:);
    #   % dd = diag(ss);
    #   % constrait (weight) is effectively less as time increases.
    #   Do2 = [sparse(length(ii),Nt) dd/param.dz -dd/param.dz]*W_OCN_DNUP;
    # else
    #   Do2 = [];
    # end

    # if exist('W_OCN_DNUP','var')
    if weights.W_OCN_DNUP:
        #   dd_dnup = speye(Nz,Nz);
        dd_dnup = sp.sparse.eye(Nz, Nz).tolil()
        #   [~,imax]=max(glider.ctd_depth);
        imax = np.argmax(glider.ctd_depth)
        #   time1 = interp_nm(glider.ctd_depth(1:imax), glider.Mtime(1:imax), gz);
        #   time2 = interp_nm(glider.ctd_depth(imax:end), glider.Mtime(imax:end), gz);
        # TODO - try straight interp for now - probably need to build interp_nm for general case
        if True:
            time1 = ADCPUtils.interp_nm(
                glider.ctd_depth[: imax + 1], glider.ctd_time[: imax + 1], gz, fill_value=np.nan
            )
            time2 = ADCPUtils.interp_nm(glider.ctd_depth[imax:], glider.ctd_time[imax:], gz, fill_value=np.nan)
        else:
            time1 = sp.interpolate.interp1d(
                glider.ctd_depth[: imax + 1],
                glider.ctd_time[: imax + 1],
                bounds_error=False,
                # fill_value="extrapolate",
                fill_value=np.nan,
            )(gz)
            time2 = sp.interpolate.interp1d(
                glider.ctd_depth[imax:],
                glider.ctd_time[imax:],
                bounds_error=False,
                # fill_value="extrapolate",
                fill_value=np.nan,
            )(gz)
        #   ss = 1-(time2-time1); % 1-diff in days
        #   ss(ss<0)=0; % limit to zero.
        #   for k=1:length(ss)
        #     dd_dnup(k,k)=ss(k);
        #   end
        ss = 1 - (time2 / 86400.0 - time1 / 86400.0)  # 1-diff in days
        ss[ss < 0] = 0  # limit to zero
        for k in range(len(ss)):
            dd_dnup[k, k] = ss[k]
        #   ii = find(ss>0 & isfinite(ss));
        #   dd_dnup=dd_dnup(ii,:);
        ii = np.nonzero(np.logical_and(ss > 0, np.isfinite(ss)))[0]
        # TODO - this should work:
        # dd_dnup = dd_dnup[ii, :]
        # But we get "IndexError: index results in >2 dimensions" - which seems to be
        # an issue with fancy indexing in sparse along multiple dimensions.  Update scipy and see if things
        # are improved.  Meantime....
        dd_dnup = sp.sparse.csr_array(np.squeeze(dd_dnup.todense()[ii, :]))
        dd_dnup.eliminate_zeros()
        #   % dd_dnup = diag(ss);
        #   % constrait (weight) is effectively less as time increases.
        #   Do2 = [sparse(length(ii),Nt) dd_dnup/param.dz -dd_dnup/param.dz]*W_OCN_DNUP;
        Do2 = (
            sp.sparse.hstack([sp.sparse.csr_array((ii.shape[0], Nt)), dd_dnup / param.dz, -dd_dnup / param.dz])
            * weights.W_OCN_DNUP
        )
        if inverse_tmp is not None:
            inverse_tmp["time1"] = time1
            inverse_tmp["time2"] = time2
            inverse_tmp["dd_dnup"] = dd_dnup.todense()
    else:
        Do2 = []
    if inverse_tmp is not None:
        inverse_tmp["Do2"] = Do2.todense()

    # % Smoothness of vehicle velocity
    # Dv = [spdiags(repmat([-1 2 -1], Nt-2,1),[0 1 2], Nt-2,Nt), sparse(Nt-2,2*Nz) ]*VEH_SMOOTH  ; % d/dt

    diags = np.tile(np.array([-1, 2, -1]), (Nt - 2, 1))
    dd_dv = sp.sparse.diags_array(diags.T, offsets=[0, 1, 2], shape=(Nt - 2, Nt))
    Dv = sp.sparse.hstack([dd_dv, sp.sparse.csr_array((Nt - 2, 2 * Nz))]) * weights.VEH_SMOOTH  # d/dt
    if inverse_tmp is not None:
        inverse_tmp["Dv"] = Dv.todense()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %% %%EXTRA CONSTRAINTS

    # % % Make the vehicle speed close to the model solution
    # %       THIS is pretty unstable, because there isn't much contraining the
    # %       low-frequency ocean velocity. The ADCP data is all about the
    # %       relative velocity...
    # %
    # % However, I use this to make the vehicle speed close to the fligth model
    # % solution if I don't have ADCP data there (one-way profile)

    # if exist('W_MODEL','var')

    #   ii_no_adcp = find(isnan(mean(real(D.UV),1,'omitnan')));
    #   G_model = sparse(length(ii_no_adcp),Nt+2*Nz);
    #   for kk=1:length(ii_no_adcp)
    #     G_model(kk,ii_no_adcp(kk))=1;
    #   end
    #   d_model = W_MODEL*transpose(D.UVttw_model(ii_no_adcp));
    # else
    #   G_model = [];
    #   d_model = [];
    # end

    if weights.W_MODEL:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ii_no_adcp = np.nonzero(np.isnan(np.nanmean(np.real(D.UV), axis=0)))[0]
        G_model = sp.sparse.lil_array((ii_no_adcp.shape[0], Nt + 2 * Nz))
        for kk in range(ii_no_adcp.shape[0]):
            G_model[kk, ii_no_adcp[kk]] = 1
        G_model = G_model.tocsr()
        d_model = D.UVttw_model[ii_no_adcp] * weights.W_MODEL
    else:
        G_model = []
        d_model = []

    if inverse_tmp is not None:
        inverse_tmp["G_model"] = G_model.todense()
        inverse_tmp["d_model"] = d_model

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Make deep ocean velocity small...
    # % That can be useful to make the velocity profiles look nice, to get rid of
    # % large mean shear bias.

    # if exist('W_deep','var')
    #   ii = find(gz>W_deep_z0); % depths above the place were we minimize deep velocities
    #   tmp = hanning(length(ii)*2);
    #   ww = [tmp(1:length(ii)) ; ones(Nz-length(ii),1)];
    #   dd = speye(Nz,Nz);
    #   for k=1:length(ww)
    #     dd(k,k)=ww(k);
    #   end
    #   dd=dd(ii,:);
    #   G_deep = [sparse(length(ii),Nt) dd dd]*W_deep;
    #   d_deep=zeros(size(G_deep,1),1);
    # else
    #   G_deep=[];
    #   d_deep=[];
    # end

    if weights.W_deep and weights.W_deep_z0:
        # TODO - untested
        ii = np.nonzero(gz > weights.W_deep_z0)[0]  # depths above the place were we minimize deep velocities
        tmp = np.hanning(ii.shape[0] * 2)
        ww = np.hstack([tmp[np.arange(ii.shape[0])], np.ones(Nz - ii.shape[0])])
        dd = sp.sparse.eye(Nz, Nz).tolil()
        for k in range(ww.shape[0]):
            dd[k, k] = ww[k]
        dd = dd[ii, :].tocsr()
        G_deep = sp.sparse.hstack([sp.sparse.csr_array((ii.shape[0], Nt)), dd, dd]) * weights.W_deep
        d_deep = np.zeros(G_deep.shape[0])
    else:
        G_deep = []
        d_deep = []

    if inverse_tmp is not None:
        inverse_tmp["G_deep"] = G_deep.todense()
        inverse_tmp["d_deep"] = d_deep
    # Good to here
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # % % Make the ttw speed zero at the bottom (when the flight model says it's zero)
    # if exist('W_MODEL_bottom','var')
    #   ii = find(D.UVttw_model==0 & D.Z0>0.5*max(D.Z0));
    #   G_bottom = zeros(length(ii), size(G_adcp,2));
    #   for k=1:length(ii)
    #     G_bottom(k,ii(k))=W_MODEL_bottom;
    #   end
    #   d_bottom = zeros(length(ii),1);
    # else
    #   G_bottom = [];
    #   d_bottom = [];
    # end

    if weights.W_MODEL_bottom:
        ii = np.nonzero(np.logical_and(D.UVttw_model == 0, D.Z0 > 0.5 * np.max(D.Z0)))[0]  # noqa: SIM300
        G_bottom = np.zeros(ii.shape[0] * G_adcp.shape[1]).reshape((ii.shape[0], G_adcp.shape[1]))
        for k in range(ii.shape[0]):
            G_bottom[k, ii[k]] = weights.W_MODEL_bottom
        d_bottom = np.zeros(ii.shape[0])
    else:
        G_bottom = []
        d_bottom = []

    if inverse_tmp is not None:
        inverse_tmp["G_bottom"] = G_bottom
        inverse_tmp["d_bottom"] = d_bottom

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %% so we have a LSQ problem
    # % [H;D;Dv]*M-[d;0;0] = 0

    # M = [G_adcp;G_dac;G_sfc;Do;Do2;Dv;G_model;G_deep;G_bottom]\
    # [d_adcp;d_dac;d_sfc;zeros(size(Do,1)+size(Do2,1)+size(Dv,1),1);d_model; d_deep; d_bottom];

    # Iterate through the A matrix, eliminating any empty list entries
    A_vars = [G_adcp, G_dac, G_sfc, Do, Do2, Dv, G_model, G_deep, G_bottom]
    A_list = []
    for ii in range(len(A_vars)):
        if isinstance(A_vars[ii], list):
            continue
        else:
            A_list.append(A_vars[ii])

    # Iterate through the B matrix, eliminating any empty list entries and converting
    # everything to np.ndarrays (needed to the call to lsqr()
    B_vars = [
        d_adcp,
        d_dac,
        d_sfc,
        np.zeros(Do.shape[0] + Do2.shape[0] + Dv.shape[0]),
        d_model,
        d_deep,
        d_bottom,
    ]
    B_list = []
    for ii in range(len(B_vars)):
        if isinstance(B_vars[ii], list):
            continue
        elif sp.sparse.issparse(B_vars[ii]):
            B_list.append(np.squeeze(B_vars[ii].todense()))
        else:
            B_list.append(B_vars[ii])

    A = sp.sparse.vstack(A_list).tocsr()
    B = np.hstack(B_list)
    lsqr_ret = sp.sparse.linalg.lsqr(A, B, show=False)
    M = lsqr_ret[0]

    # UVttw = transpose(M(1:Nt)); % solved-for TTW component
    UVttw = M[:Nt]  # solved-for TTW component
    if inverse_tmp is not None:
        inverse_tmp["UVttw"] = UVttw

    # UVocn = transpose(M(Nt+1:end));
    # UVocn = reshape(UVocn,[],2);
    # TODO - Luc - Reshape needed for profile below
    UVocn = M[Nt:].reshape(-1, 2, order="F")

    if inverse_tmp is not None:
        inverse_tmp["UVocn"] = UVocn

    # % INVERSE SOLUTION

    # % U_drift = Ocean velocity at the glider
    # D.UVocn_solution = transpose(Ai0*UVocn(:));
    D.UVocn_solution = Ai0 * np.squeeze(UVocn.reshape(1, -1, order="F"))

    # % Glider speed through the water
    # D.UVttw_solution = UVttw
    D.UVttw_solution = UVttw

    # % Total vehicle speed: U_ttw (speed through the water) + U_drift (ocean speed at the glider).
    # D.UVveh_solution =  UVttw+transpose(Ai0*UVocn(:));
    D.UVveh_solution = UVttw + (Ai0 * np.squeeze(UVocn.reshape(1, -1, order="F")))

    # % ADCP measurement (D.UV) = u_ocean(t,z) - u_drift( u_ocean @ glider) - u_ttw(@ glider)
    # % Ocean velocity at the measurement location:
    # D.UVocn_adcp= D.UV+D.UVttw_solution + D.UVocn_solution; % Measured ocean velocity measurements
    D.UVocn_adcp = D.UV + D.UVttw_solution + D.UVocn_solution  # Measured ocean velocity measurements

    # D.UVerr = D.UVocn_adcp-D.UVocn_solution;
    D.UVerr = D.UVocn_adcp - D.UVocn_solution

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%  Vertical velocity inverse

    # Gv_adcp = W_MEAS*[-Av, AiM-Av*Ai0];  % same as the horizontal velocity.
    Gv_adcp = sp.sparse.hstack([-Av, AiM - Av * Ai0]) * weights.W_MEAS

    # M = [Gv_adcp;G_dac;G_sfc;2*Do;2*Do2;Dv]\[dw_adcp;0*d_dac;0*d_sfc;zeros(size(Do,1)+size(Do2,1)+size(Dv,1),1)];
    A_vars = [Gv_adcp, G_dac, G_sfc, 2 * Do, 2 * Do2, Dv]
    A_list = []
    for ii in range(len(A_vars)):
        if isinstance(A_vars[ii], list):
            continue
        else:
            A_list.append(A_vars[ii])

    # Iterate through the B matrix, eliminating any empty list entries and converting
    # everything to np.ndarrays (needed to the call to lsqr()
    # B_vars = [dw_adcp, 0 * d_dac, 0 * d_sfc, np.zeros(Do.shape[0] + Do2.shape[0] + Dv.shape[0])]
    B_vars = [
        dw_adcp,
        np.zeros(d_dac.shape[0]),
        np.zeros(d_sfc.shape[0]),
        np.zeros(Do.shape[0] + Do2.shape[0] + Dv.shape[0]),
    ]
    B_list = []
    for ii in range(len(B_vars)):
        if isinstance(B_vars[ii], list):
            continue
        elif sp.sparse.issparse(B_vars[ii]):
            B_list.append(np.squeeze(B_vars[ii].todense()))
        else:
            B_list.append(B_vars[ii])

    A = sp.sparse.vstack(A_list).tocsr()
    B = np.hstack(B_list)
    lsqr_ret = sp.sparse.linalg.lsqr(A, B, show=False)
    M = lsqr_ret[0]

    # % Glider speed through the water
    # D.Wttw_solution = transpose(M(1:Nt)); % solved-for TTW component
    # Imaginary compoents are all 0 - matches matlab
    D.Wttw_solution = M[:Nt]

    # % Ocean velocity

    # Wocn = M(Nt+1:end);
    # Wocn = reshape(Wocn,[],2);
    # TODO - Luc - Reshape needed for profile below
    # Imaginary components are all 0 - matches matlab
    Wocn = M[Nt:].reshape(-1, 2, order="F")

    # % U_drift = Ocean velocity at the glider
    # D.Wocn_solution = transpose(Ai0*Wocn(:));
    D.Wocn_solution = Ai0 * np.squeeze(Wocn.reshape(1, -1, order="F"))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %% Gridded version (on the ocean vertical grid).

    # [~,iiz, itmp] = intersect(gz, profile.z);
    _, iiz, itmp = np.intersect1d(gz, profile.z, return_indices=True)

    profile.UVocn[itmp, :] = UVocn[iiz, :]
    idn = np.nonzero(D.upcast == 0)[0]
    iup = np.nonzero(D.upcast == 1)[0]

    profile.UVttw_solution[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], D.UVttw_solution[idn], gz[iiz])[0]
    profile.UVttw_solution[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], D.UVttw_solution[iup], gz[iiz])[0]
    profile.UVttw_model[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], D.UVttw_model[idn], gz[iiz])[0]
    profile.UVttw_model[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], D.UVttw_model[iup], gz[iiz])[0]

    profile.time[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], D.time[idn], gz[iiz])[0]
    profile.time[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], D.time[iup], gz[iiz])[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile.UVerr[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], np.nanstd(D.UVerr[:, idn], 0), gz[iiz])[0]
        profile.UVerr[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], np.nanstd(D.UVerr[:, iup], 0), gz[iiz])[0]

    profile.Wttw_solution[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], D.Wttw_solution[idn], gz[iiz])[0]
    profile.Wttw_solution[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], D.Wttw_solution[iup], gz[iiz])[0]
    profile.Wttw_model[itmp, 0] = ADCPUtils.bindata(D.Z0[idn], D.Wttw_model[idn], gz[iiz])[0]
    profile.Wttw_model[itmp, 1] = ADCPUtils.bindata(D.Z0[iup], D.Wttw_model[iup], gz[iiz])[0]

    profile.Wocn[itmp, :] = Wocn[iiz, :]

    # %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # variables_for_plots.gps_constraints = gps_constraints;
    # variables_for_plots.TT=TT;
    # variables_for_plots.G_adcp = G_adcp;
    # variables_for_plots.M = M;
    # variables_for_plots.AiM = AiM;
    # variables_for_plots.AiG = AiG;
    # variables_for_plots.Av = Av;
    # variables_for_plots.Nt = Nt;
    # variables_for_plots.Nz = Nz;

    return (D, profile, None, inverse_tmp)
