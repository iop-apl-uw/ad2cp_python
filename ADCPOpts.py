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

"""ADCPOpts.py - SG ADCP processing options and command-line parsing."""

import argparse
import pathlib
import sys
from collections.abc import Sequence
from typing import Any


class FullPathlibAction(argparse.Action):
    """argparse action converting a string (or list of strings) into absolute pathlib.Path(s)."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = None,
        **kwargs: Any,  # noqa: ANN401 -- forwarded to argparse.Action.__init__'s many optional params
    ) -> None:
        """Initializes the action, forwarding to argparse.Action.

        Args:
            option_strings: Command-line option strings, e.g. ``["--foo"]``.
            dest: Namespace attribute name to store the value under.
            nargs: Number of command-line arguments to consume; unused/unchecked here.
            **kwargs: Forwarded to ``argparse.Action.__init__``.
        """
        # if nargs is not None:
        #    raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Converts the parsed value(s) to absolute pathlib.Path(s) and stores them on the namespace.

        Args:
            parser: The argument parser (unused; accepted for the ``argparse.Action`` interface).
            namespace: Namespace to update with the converted value.
            values: The raw string, list of strings, or None parsed for this argument.
            option_string: The option string used (unused; accepted for the ``argparse.Action`` interface).
        """
        if values == "" or values is None:
            setattr(namespace, self.dest, "")
        elif isinstance(values, str):
            # setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
            setattr(namespace, self.dest, pathlib.Path(values).expanduser().absolute())
        else:
            # setattr(namespace, self.dest, list(map(lambda y: os.path.abspath(os.path.expanduser(y)), values)))
            setattr(namespace, self.dest, list(map(lambda y: pathlib.Path(y).expanduser().absolute(), values)))


def ADCPOptions(description: str, calling_module: str, cmdline_args: list[str] = sys.argv) -> argparse.Namespace | None:
    """Defines and parses command line options for the standalone adcp CLI entry points.

    Args:
        description: Description shown in ``--help`` output.
        calling_module: Name of the calling module -- ``"SGADCP"`` or ``"SGADCPPlot"``,
            selects which additional options are defined.
        cmdline_args: Command line arguments to parse.

    Returns:
        Parsed options, or None if ``calling_module`` isn't recognized.
    """
    if calling_module not in ("SGADCP", "SGADCPPlot"):
        sys.stderr.write(f"Unknown calling_module {calling_module}")
        return None

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")
    ap.add_argument("--debug", default=False, action="store_true", help="enable debug output")
    ap.add_argument(
        "--adcp_log",
        help="Name of output logfile",
        action=FullPathlibAction,
    )
    ap.add_argument(
        "--plot_directory",
        help="Override default plot directory location",
        action=FullPathlibAction,
        default=None,
    )
    ap.add_argument(
        "--debug_pdb",
        help="Enter the debugger for selected exceptions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    if calling_module == "SGADCP":
        ap.add_argument(
            "--mission_dir",
            help="Directory where profile netcdfs are located",
            action=FullPathlibAction,
            required=True,
        )

        ap.add_argument("--adcp_config_file", help="Configuration input file", action=FullPathlibAction, default="")

        ap.add_argument(
            "--save_details",
            help="Save detailed variables for later comparison",
            action="store_true",
            default=False,
        )

        ap.add_argument(
            "--var_meta_filename",
            help="ADCP variable metadata configiuration YAML file",
            action=FullPathlibAction,
            default=pathlib.Path(__file__).parent.joinpath("config/var_meta.yml"),
        )

        ap.add_argument(
            "--global_meta_filename",
            help="ADCP global metadata configiuration YAML file",
            action=FullPathlibAction,
            default=None,
        )

        ap.add_argument(
            "--include_glider_vars",
            help="Includes a selection of varibles from the Seglider CTD (used for plotting)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        ap.add_argument("ncf_files", help="Seaglider netcdf files", nargs="*")

    if calling_module == "SGADCPPlot":
        ap.add_argument("ncf_filename", help="Seaglider ADCP processing output netcdf file", action=FullPathlibAction)
        ap.add_argument("--min_plot_depth", help="Minimum depth to plot", type=float, default=0.0)
        ap.add_argument("--max_plot_depth", help="Maximum depth to plot", type=float, default=1000.0)

    args = ap.parse_args(cmdline_args)

    return args
