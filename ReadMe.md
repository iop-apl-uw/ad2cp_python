# Seaglider Microstructure Processing

Shore side processing software for the microstructure system used 
on the [Seaglider](https://iop.apl.washington.edu/seaglider.php) buoyancy driven
autonomous underwater vehicles, developed at the University of Washington,
maintained by the [IOP group at APL-UW](https://iop.apl.washington.edu/index.php) and 
[Rockland Scientific](https://rocklandscientific.com/).

# Use of this code

This code base can be run as part of the Seaglider basestation processing
system, or it can be run independently.

# Current status

Only processing of the log averaged data collected and transmitted back to the Seaglider
basestation by the TMicro system is currently supported.  Future work will extend this to processing
the raw data collected and left on the microstructure system's sd card.

# Overview

# Output products

- Main approach is to update the Seaglider per-dive netcdf file (pXXXYYY.nc - where XXX is 
glider id and YYYY is dive number) with the scientifically interesting values (chi and epsilon) 
and possibly other intermediate calculations.  Since that netcdf file is quite large, separate processing
(ie. SimpleNetCDF.py extension from the basestation) is relied on to a more manageable output 
of real-time data.

- When process of the raw data comes on-line, the option to update the per-dive netcdf file will remain,
but something else might be preferable (such as a netcdf/matfile with just the microstructure columns).

- The need to see an entire mission of data is, for now, handled by MakeMissionTimeseries.py from the 
Seaglider basestation.
