# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-07-23

### Added

- Support for reading "logdev" ADCP files (`cp_*` variable prefix), in
  addition to the existing `ad2cp_*` real-time files.
- Support for the ADCP `coordinateSystem` variable: ENU and XYZ instrument
  coordinate systems are now handled explicitly in the earth/instrument
  frame transform, and beam-coordinate data is rejected with a clear error
  instead of being silently mishandled.
- Optional-variable handling in the netCDF name-mapping table, so files
  missing non-essential variables (e.g. `coordinateSystem`) no longer fail
  to load.

### Changed

- Switched command-line path arguments from a custom `FullPathAction` to
  `FullPathlibAction`, returning `pathlib.Path` objects instead of strings,
  in line with the project's `pathlib`-first conventions.
- Updated type hints and fixed a plotting function signature in
  `DiveOcean.py` and `MissionOcean.py`.

### Fixed

- Satisfied ruff lint checks on the new ADCP file-loading code.

## [0.0.1] - 2026-05-09

Initial release. This version ports the MATLAB ADCP inversion processing
(originally authored by Luc Rainville) to Python and integrates it with the
Seaglider basestation3 pipeline.

### Added

- Core ADCP inversion processing: earth/instrument frame conversion, sparse
  least-squares inversion (`bindata_AS`, `interp_nm`), and profile gridding,
  ported from the MATLAB reference implementation and validated against it.
- Real-time ADCP data processing from data uploaded over the Iridium
  satellite link.
- Stand-alone processing of the complete ADCP dataset recovered from the
  Seaglider's scicon (`SGADCP.py`).
- Downward-looking ADCP support, in addition to the upward-looking case.
- Basestation3 extensions:
  - `BaseADCP.py` — per-dive processing, invoked via the `[postnetcdf]`
    extension hook.
  - `BaseADCPMission.py` — whole-mission netCDF file generation, invoked via
    the `[missionearly]` extension hook.
- Output products:
  - Ocean and glider velocity timeseries and gridded profile variables
    written into the Seaglider per-dive netCDF file (see the Output columns
    tables in `ReadMe.md`).
  - Whole-mission timeseries and gridded netCDF files for stand-alone runs.
- Plotting:
  - `DiveOcean.py` — per-dive ocean velocity vs. depth plots, with isopycnal
    overlays and configurable colormap ranges.
  - `SGADCPPlot.py` — stand-alone plotting driver.
  - `MissionOcean.py` — whole-mission ocean velocity section plots, driven
    by `sections.yml`.
  - Experimental 3D ocean velocity plot (behind a feature switch).
- Configuration:
  - `var_meta.yml` for netCDF variable metadata, validated with pydantic
    models.
  - ADCP processing configuration file, with documentation.
  - Support for ADCP pressure sensor option, `MAX_TILT`, `WMAX_error`, and
    `UMAX` as configurable parameters.
  - `--debug_pdb` command-line option for interactive debugging.
- Test suite (pytest) and CI pipeline (ruff, mypy, GitHub Actions) covering
  `SGADCP.py` and supporting modules.
- Documentation: installation and usage instructions in `ReadMe.md`, sample
  `sections.yml`, and the "Seaglider ADCP Processing" reference paper
  (Rainville et al. 2024) under `docs/`.

### Known limitations

- Only real-time data processing is supported when run as part of the
  basestation3 pipeline; the stand-alone path does not yet save all output
  products for the full ADCP dataset.
- Downward-looking ADCP processing is not yet fully correct; a fix is in
  progress.
