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
SGADCP.py - main entry point for stand alone processing of the Seaglider
ADCP data
"""

import os
import pathlib
import pdb
import sys
import time
import traceback

import h5py

# import netCDF4
import numpy as np

import ADCP
import ADCPConfig
import ADCPFiles
import ADCPOpts
import ADCPRealtime
import ADCPUtils
from ADCPLog import ADCPLogger, log_critical, log_debug, log_error, log_info

DEBUG_PDB = True


def DEBUG_PDB_F() -> None:
    """Enter the debugger on exceptions"""
    if DEBUG_PDB:
        _, __, traceb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(traceb)


def main() -> int:
    """Command line driver Seaglider ADCP data processing

    Returns:
        0 for success
        Non-zero for critical problems.

    Raises:
        Any exceptions raised are considered critical errors and not expected
    """

    # NOTE: Minimum options here - debug level and name for yaml config file

    # Get options
    adcp_opts = ADCPOpts.ADCPOptions("Command line driver Seaglider ADCPx data processing")

    # Initialize log
    ADCPLogger(adcp_opts, include_time=True)

    if ADCPUtils.check_versions():
        return 1

    # Read the config file
    param, weights = ADCPConfig.ProcessConfigFile(adcp_opts.adcp_config_file)
    if not param:
        return 1

    # param.gz = (0:param.dz:1000)';
    # Done in ADCPConfig

    # Mission-long corrections - not used in the 2024 version of the real-time code
    # if exist('dpitch','var')
    # param.dpitch = dpitch;
    # end
    # if exist('dpitch','var')
    # param.droll = droll;
    # end
    # if exist('dpitch','var')
    # param.dheading = dheading;
    # end

    log_debug(param)

    # Whole mission accumulators
    depth_grid = None
    mission_dive = []
    mission_time = []
    mission_uocn = []
    mission_vocn = []
    mission_wocn = []

    # adcp_grid.UVttw_model(iiz,n_dive*2-1:n_dive*2) = profile.UVttw_model(itmp,:);
    # adcp_grid.UVerr(iiz,n_dive*2-1:n_dive*2) = profile.UVerr(itmp,:);

    # adcp_grid.Wocn(iiz,n_dive*2-1:n_dive*2) = profile.Wocn(itmp,:);
    # adcp_grid.Wttw_solution(iiz,n_dive*2-1:n_dive*2) = profile.Wttw_solution(itmp,:);
    # adcp_grid.Wttw_model(iiz,n_dive*2-1:n_dive*2) = profile.Wttw_model(itmp,:);

    # % NaN the edges
    # adcp_grid.time(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVocn(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVttw_solution(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVttw_model(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.UVerr(i0z,n_dive*2-1:n_dive*2) = NaN;

    # adcp_grid.Wocn(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.Wttw_solution(i0z,n_dive*2-1:n_dive*2) = NaN;
    # adcp_grid.Wttw_model(i0z,n_dive*2-1:n_dive*2) = NaN;

    # % from the glider profile
    # [~,imax] = max(glider.ctd_depth);
    # adcp_grid.lon(1,n_dive*2-1)=mean(glider.longitude(1:imax));
    # adcp_grid.lon(1,n_dive*2  )=mean(glider.longitude(imax:end));
    # adcp_grid.lat(1,n_dive*2-1)=mean(glider.latitude(1:imax));
    # adcp_grid.lat(1,n_dive*2  )=mean(glider.latitude(imax:end));

    output_filename = None

    # TODO - load template for header meta data and override template

    var_meta = ADCPConfig.LoadVarMeta(adcp_opts.adcp_var_meta_filename)

    dive_nc_filenames: list[pathlib.Path] = []

    if adcp_opts.ncf_files:
        for ff in adcp_opts.ncf_files:
            dive_nc_filenames.append(adcp_opts.mission_dir / ff)
    else:
        for m in adcp_opts.mission_dir.glob("p[0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc"):
            dive_nc_filenames.append(m)
    dive_nc_filenames = sorted(dive_nc_filenames)

    for ncf_name in dive_nc_filenames:
        log_info(f"Processing {ncf_name}")
        ds = ADCPUtils.open_netcdf_file(ncf_name)
        if ds is None:
            continue

        if not param.sg:
            param.sg = ds.glider

        if not output_filename:
            output_filename = pathlib.Path(adcp_opts.mission_dir).joinpath(
                f"{ADCPUtils.GetMissionStr(ds).replace(' ', '-').replace('_', '-')}-adcp-realtime.nc"
            )

        # Read real-time
        try:
            glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, ncf_name, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem loading data", "exc")
            continue
        finally:
            ds.close()

        param.time_limits = np.array((np.min(gps.log_gps_time), np.max(gps.log_gps_time)))

        # Transform velocites to instrument frame
        try:
            ADCPRealtime.TransformToInstrument(adcp_realtime)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem transforming compass data", "exc")
            continue

        # Clean up adcp data
        try:
            ADCP.CleanADCP(adcp_realtime, glider, param)
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem cleaning realtime adcp data", "exc")
            continue

        # From matlab code - caclulated but unused
        # time_diff = np.diff(adcp_realtime.time)
        # dt = np.max([15.0, np.median(time_diff[time_diff > 2.0])])

        # Perfrom inverse
        try:
            D, profile, variables_for_plot, inverse_tmp = ADCP.Inverse(
                adcp_realtime, gps, glider, weights, param, {} if adcp_opts.save_details else None
            )
        except Exception:
            DEBUG_PDB_F()
            log_error("Problem performing inverse calculation", "exc")
            continue

        mission_dive.append(glider.dive)
        mission_dive.append(glider.dive)
        mission_time.append(profile.time)
        mission_uocn.append(profile.UVocn.real)
        mission_vocn.append(profile.UVocn.imag)
        mission_wocn.append(profile.Wocn)
        if depth_grid is None:
            depth_grid = profile.z

        # Debug save out - for comparision with other processing
        # TODO - need a better name - match with input file
        if adcp_opts.save_details:
            with h5py.File("test.hdf5", "w") as hdf:
                adcp_realtime.save_to_hdf5("adcp_realtime", hdf)
                glider.save_to_hdf5("glider", hdf)
                gps.save_to_hdf5("gps", hdf)
                D.save_to_hdf5("D", hdf)
                profile.save_to_hdf5("profile", hdf)
                grp = hdf.create_group("inverse_tmp")
                for k, v in inverse_tmp.items():
                    grp.create_dataset(k, data=v)

    # Output results

    # TODO - CONSIDER - trim arrays based on deepest actual observation

    # Header templates
    ad2cp_variable_mapping = {
        # ADCP profile variables
        "inverse_profile_depth": depth_grid,
        "inverse_profile_dive": np.hstack(mission_dive),
        "inverse_profile_time": np.hstack(mission_time),
        "inverse_profile_velocity_north": np.hstack(mission_uocn),
        "inverse_profile_velocity_east": np.hstack(mission_vocn),
        "inverse_profile_velocity_vertical": np.hstack(mission_wocn),
    }

    dso = ADCPUtils.open_netcdf_file(output_filename, "w")
    if dso is None:
        log_error(f"Could not open {output_filename}")
        return 1

    try:
        ADCPUtils.CreateNCVars(dso, ad2cp_variable_mapping, var_meta)
    except Exception:
        DEBUG_PDB_F()
        log_error(f"Problem stripping creating variables in {output_filename}", "exc")
    finally:
        dso.sync()
        dso.close()

    # TODO - apply global attributes

    return 0


if __name__ == "__main__":
    # Force to be in UTC
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        ret_val = main()
    except Exception:
        DEBUG_PDB_F()
        log_critical("Unhandled exception in main -- exiting", "exc")

    sys.exit(ret_val)
