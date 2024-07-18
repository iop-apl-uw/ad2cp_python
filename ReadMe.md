# Seaglider Nortek ADCP Processing

Shore side processing software for the ADCP system used 
on the [Seaglider](https://iop.apl.washington.edu/seaglider.php) buoyancy driven
autonomous underwater vehicles, developed at the University of Washington,
maintained by the [IOP group at APL-UW](https://iop.apl.washington.edu/index.php) and 
[Nortek Group](https://www.nortekgroup.com/).

# Use of this code

This code base can be run as part of the Seaglider basestation processing
system - to process the real-time ADCP data uploaded over the iridium sat link, 
or it can be run independently - to process either the real-time data, or the complete
ADCP dataset that is available on the Seaglider's scicon after recovery.

# Current status

Only processing of the real-time data is supported at this time.

# Overview
*Insert sketch of processing here*

# Output products

- When this code is run from a Seaglider Basestation, the Seaglider per-dive netcdf file (pXXXYYY.nc - where XXX is 
glider id and YYYY is dive number) is updated with the scientifically interesting values (ocean velocity)
and possibly other intermediate calculations.  Since that netcdf file is quite large, separate processing
(ie. SimpleNetCDF.py extension from the basestation) is relied on to a more manageable output 
of real-time data.

- When this code is run stand-alone, for either the real-time data or the full ADCP data, the output is loaded into netcdf
files - one for a timeseries and a separate one for a grided product. Any subset of the mission may be processed, but typically, this path processes the entire mission's data set.

# Output columns 

*Add documentation for each column*

D.time
D.z0
D.UVocn_solution
D.UVttw_solution
D.Wttw_solution
D.Wocn_solution

profile.z
profile.UVocn
profile.Wocn

# Note on code structure

This code is directly derived from the  matlab version *insert link here*.  The overall structure 
and naming conventions are kept the same to help with readability and insight.
