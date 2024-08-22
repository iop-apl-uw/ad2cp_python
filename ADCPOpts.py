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
ADCPOpts.py - SG ADCP processing options and command-line parsing
"""

import argparse
import os
import pathlib
from typing import Any, Sequence


class FullPathAction(argparse.Action):
    def __init__(self, option_strings: Any, dest: Any, nargs: Any = None, **kwargs: Any) -> None:
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
        if values == "" or values is None:
            setattr(namespace, self.dest, "")
        elif isinstance(values, str):
            # setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
            setattr(namespace, self.dest, pathlib.Path(values).expanduser().absolute())
        else:
            # setattr(namespace, self.dest, list(map(lambda y: os.path.abspath(os.path.expanduser(y)), values)))
            setattr(namespace, self.dest, list(map(lambda y: pathlib.Path(y).expanduser().absolute(), values)))


def ADCPOptions(description: str) -> argparse.Namespace:
    app_name, _ = os.path.splitext(__file__)

    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--verbose", default=False, action="store_true", help="enable verbose output")
    ap.add_argument("--debug", default=False, action="store_true", help="enable debug output")
    ap.add_argument(
        "--adcp_log",
        help="Name of output logfile",
        action=FullPathAction,
    )
    ap.add_argument(
        "--mission_dir",
        help="Directory where profile netcdfs are located",
        action=FullPathAction,
        required=True,
    )
    ap.add_argument("--adcp_config_file", help="Configuration input file", action=FullPathAction, default="")

    ap.add_argument(
        "--save_details",
        help="Save detailed variables for later comparison",
        action="store_true",
        default=False,
    )

    ap.add_argument(
        "--var_meta_filename",
        help="ADCP variable metadata configiuration YAML file",
        action=FullPathAction,
        default=pathlib.Path(__file__).parent.joinpath("config/var_meta.yml"),
    )

    ap.add_argument(
        "--global_meta_filename",
        help="ADCP global metadata configiuration YAML file",
        action=FullPathAction,
        default=None,
    )

    ap.add_argument("ncf_files", help="Seaglider netcdf files", nargs="*")

    args = ap.parse_args()

    return args
