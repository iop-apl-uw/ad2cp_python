# -*- python-fmt -*-

## Copyright (c) 2024, 2025  University of Washington.
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


import os
import pathlib
import shutil
import time
from collections.abc import Callable

import pytest

# Each test in a "mission_dir" under the testdata/XXXX directory - testdata/sg179_Guam_Oct19/mission_dir for example
# Previous runs are removed and the contents of testdata/XXXX (no sub-directories) are copied to
# testdata/XXXX/mission_dir


def run_mission(
    data_dir: pathlib.Path,
    mission_dir: pathlib.Path,
    main_func: Callable[[list[str]], int],
    cmd_line: list[str],
    caplog: pytest.LogCaptureFixture,
    allowed_msgs: list[str],
) -> None:
    """Copies a mission to a test directory, executes it and checks warning and error output against a known list.

    Args:
        data_dir: Original data directory to copy from (testdata/XXXX).
        mission_dir: Test mission directory to populate and run in; wiped first if it exists.
        main_func: The main() toplevel entry point called from __main__ - takes a command line argument.
        cmd_line: Argument list to pass to main_func.
        caplog: Logging capture from the pytest ``caplog`` fixture.
        allowed_msgs: List of allowed messages that can appear in the caplog without failing the test.
    """
    os.environ["TZ"] = "UTC"
    time.tzset()

    # Clean up previous run - if any
    if mission_dir.exists():
        shutil.rmtree(mission_dir)
    # Populate the new mission_dir
    mission_dir.mkdir()
    for p in data_dir.iterdir():
        if p.is_dir():
            continue
        shutil.copy(p, mission_dir)
    result = main_func(cmd_line)
    assert result == 0
    bad_errors = ""
    for record in caplog.records:
        # Check for known WARNING, ERROR or CRITICAL msgs
        for msg in allowed_msgs:
            if msg in record.msg:
                break
        else:
            if record.levelname in ["CRITICAL", "ERROR", "WARNING"]:
                bad_errors += f"{record.levelname}:{record.getMessage()}\n"
    if bad_errors:
        # pytest.fail's @_with_exception decorator confuses ty's overload resolution
        pytest.fail(bad_errors)  # ty: ignore[invalid-argument-type]
