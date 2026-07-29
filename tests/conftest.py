# -*- python-fmt -*-

## Copyright (c) 2026  University of Washington.
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

import logging
from collections.abc import Generator

import pytest

import ADCPLog


@pytest.fixture(autouse=True)
def _reset_adcp_logger() -> Generator[None]:
    """Resets ADCPLogger's process-global state before and after every test.

    ADCPLogger.is_initialized/.log/.opts/.debug_enabled/.info_enabled are class
    attributes (singleton-style), so any test or code-under-test (e.g.
    SGADCP.main()) that constructs an ADCPLogger leaks its config into every
    later test in the process unless reset here. The log_*_max_count
    defaultdicts have the same whole-process lifetime and are cleared for the
    same reason.
    """
    _reset()
    yield
    _reset()


def _reset() -> None:
    if ADCPLog.ADCPLogger.log is not None:
        for handler in list(ADCPLog.ADCPLogger.log.handlers):
            ADCPLog.ADCPLogger.log.removeHandler(handler)
            handler.close()
    warnings_logger = logging.getLogger("py.warnings")
    for handler in list(warnings_logger.handlers):
        warnings_logger.removeHandler(handler)
        handler.close()

    ADCPLog.ADCPLogger.self = None
    ADCPLog.ADCPLogger.is_initialized = False
    ADCPLog.ADCPLogger.opts = None
    ADCPLog.ADCPLogger.log = None
    ADCPLog.ADCPLogger.stringHandler = None
    ADCPLog.ADCPLogger.stringBuffer = None
    ADCPLog.ADCPLogger.debug_enabled = False
    ADCPLog.ADCPLogger.info_enabled = False

    ADCPLog.log_error_max_count.clear()
    ADCPLog.log_warning_max_count.clear()
    ADCPLog.log_info_max_count.clear()
    ADCPLog.log_debug_max_count.clear()
