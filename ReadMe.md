# Seaglider Nortek ADCP Processing

Shore side processing software for the Nortek Signature 1000 ADCP system ([Nortek Group](https://www.nortekgroup.com/)) used 
on the [Seaglider](https://iop.apl.washington.edu/seaglider.php) buoyancy driven
autonomous underwater vehicles, developed at the University of Washington,
maintained by the [IOP group at APL-UW](https://iop.apl.washington.edu/index.php)

# Use of this code

This code base can be run as part of the Seaglider basestation processing
system - to process the real-time ADCP data uploaded over the iridium sat link, 
or it can be run independently - to process either the real-time data, or the complete
ADCP dataset that is available on the Seaglider's scicon after recovery.

# Current status

Only processing of the real-time data is supported at this time. The stand alone processing
does not currently save any meaningful output products.

# Note on code structure

This code is directly derived from the matlab version authored by Luc Rainville.  The overall structure 
and naming conventions are kept the same to help with readability and insight.  At some point in the near 
future, we will have the matlab version available for use.

# Overview
*Insert link to "Seaglider ADCP Processing" here*

# Output products

- When this code is run from a Seaglider Basestation, the Seaglider per-dive netcdf file (pXXXYYY.nc - where XXX is 
glider id and YYYY is dive number) is updated with the scientifically interesting values (ocean velocity)
and possibly other intermediate calculations.  Since that netcdf file is quite large, separate processing
(ie. SimpleNetCDF.py extension from the basestation) is relied on to a more manageable output 
of real-time data.

- When this code is run stand-alone, for either the real-time data or the full ADCP data, the output is loaded into netcdf
files - one for a timeseries and a separate one for a grided product. Any subset of the mission may be processed, but typically, this path processes the entire mission's data set.

# Output columns 

The following tables map the internal code variables into the Seaglider per-dive netCDF file variables.  Further details 
for each are available in the "Seaglider ADCP Processing" (Rainville et al. 2024) and the metadata associated with each varaible - see [Meta data definitions](var_meta.yml).

## These are timeseries variables 
| Description                                | Program variable   | per-dive netCDF variable(s)                      |
|:-------------------------------------------|:-------------------|:-------------------------------------------------|
| epoch time for each point                  | `D.time`           | `ad2cp_inv_glider_time`                          |
| Depth of glider                            | `D.z0`             | `ad2cp_inv_glider_depth`                         |
| Ocean velocity at the glider               | `D.UVocn_solution` | `ad2cp_inv_glider_uocn`, `ad2cp_inv_glider_vocn` |
| Glider velocity through the water          | `D.UVttw_solution` | `ad2cp_inv_glider_uttw`, `ad2cp_inv_glider_vttw` |
| Vertical glider velocity through the water | `D.Wttw_solution ` | `ad2cp_inv_glider_wttw`                          |
| Vertical ocean velocity at glider          | `D.Wocn_solution`  | `ad2cp_inv_glider_wocn`                          |


## These variables are gridded onto a regular vertical grid

| Description                                  | Program variable | per-dive netCDF variable(s)                        |
|:---------------------------------------------|:-----------------|:---------------------------------------------------|
| Depth of the profile grid                    | `profile.z`      | `ad2cp_inv_profile_point`                          |
| Ocean velocity at the profile depth          | `profile.UVocn`  | `ad2cp_inv_profile_uocn`, `ad2cp_inv_profile_vocn` |
| Vertical ocean velocity at the profile depth | `profile.Wocn`   | `ad2cp_inv_profile_wocn`                           |

