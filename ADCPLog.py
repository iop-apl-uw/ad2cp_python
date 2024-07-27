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
ADCPLog.py - Logging infrastructure.

Designed to be a subeset of the logging support in the Seaglider Basestation
"""

import argparse
import collections
import inspect
import logging
import os
import sys
import traceback
from typing import DefaultDict

_stack_options = ["caller", "caller", "caller", "caller", "exc"]  # default

if "BaseLog" in sys.modules:
    raise SystemError("Internal error - ADCPLog should not be included")


class ADCPLogger:
    """
    ADCPLog: for use by all basestation code and utilities
    """

    self = None  # the global instance
    is_initialized = False
    opts = None  # whatever starting options
    log = None  # the logger
    # for transient string logging
    stringHandler = None
    stringBuffer = None
    # Turns out that calling the logging calls is expensive
    # it always generates its own line info for a logging record before
    # finally handing it off to each handler
    # But we know what we want so record locallyand use to short-circuit these calls

    # warnings, errors, and criticals are always enabled
    # -v turns on log_info, ---debug turns on log_debug
    debug_enabled = info_enabled = False
    debug_loc, info_loc, warning_loc, error_loc, critical_loc = _stack_options

    def __init__(self, opts: argparse.Namespace, include_time: bool = False) -> None:
        """
        Initializes a logging.Logger object, according to options (opts).
        """

        if not ADCPLogger.is_initialized:
            ADCPLogger.self = self
            ADCPLogger.opts = opts

            calling_module = os.path.splitext(os.path.split(inspect.stack()[1].filename)[1])[0]

            # create logger
            ADCPLogger.log = logging.getLogger(calling_module)
            assert ADCPLogger.log is not None
            ADCPLogger.log.setLevel(logging.DEBUG)

            # create a file handler if log filename is specified in opts
            if opts is not None and opts.adcp_log is not None and opts.adcp_log != "":
                fh = logging.FileHandler(opts.adcp_log)
                # fh.setLevel(logging.NOTSET) # messages of all levels will be recorded
                self.setHandler(fh, opts, include_time)

            # always create a console handler
            sh = logging.StreamHandler()
            self.setHandler(sh, opts, include_time)

            # Catch warning, error and critical

            ADCPLogger.is_initialized = True
            log_info("Process id = %d" % os.getpid())  # report our process id

    def setHandler(self, handle: logging.Handler, opts: argparse.Namespace, include_time: bool) -> None:
        """
        Set a logging handle.
        """
        if include_time:
            formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s", "%H:%M:%S %d %b %Y %Z")
        else:
            # Remove timestamps for easier log comparison and reading
            formatter = logging.Formatter("%(levelname)s: %(message)s")

        if opts is not None:
            if opts.debug:
                ADCPLogger.debug_enabled = True
                ADCPLogger.info_enabled = True
                handle.setLevel(logging.DEBUG)
            elif opts.verbose:
                ADCPLogger.info_enabled = True
                handle.setLevel(logging.INFO)
            else:
                handle.setLevel(logging.WARNING)

        else:
            handle.setLevel(logging.WARNING)

        handle.setFormatter(formatter)
        assert ADCPLogger.log is not None
        ADCPLogger.log.addHandler(handle)

        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.addHandler(handle)

    def getLogger(self) -> logging.Logger:
        """getLogger: access function to log (static member)"""
        if not ADCPLogger.is_initialized:
            # error condition
            pass

        assert ADCPLogger.log is not None
        return ADCPLogger.log


def __log_caller_info(s: object, loc: str | None) -> str:
    """Add stack or module: line number info for log caller to given string
    Input:
    s - string to be logged

    Return:
    string with possible location information added
    """
    s = str(s)
    if loc:
        try:
            # skip our local callers
            offset = 3
            # __log_caller_info(); log_XXXX; <caller>
            if loc in ["caller", "parent"]:
                if loc == "parent":  # A utlity routine
                    offset = offset + 1
                frame = traceback.extract_stack(None, offset)[0]
                module, lineno, function, _ = frame
                module = os.path.basename(module)  # lose extension
                s = "%s(%d): %s" % (module, lineno, s)
            elif loc == "exc":
                exc = traceback.format_exc()
                if exc:  # if no exception, nothing added
                    s = f"{s}:\n{exc}"
            elif loc == "stack":
                # frame_num = 0
                prefix = ">"
                stack = ""
                frames = traceback.extract_stack()
                # normally from bottom to top; reverse this so most recent call first
                frames.reverse()
                for frame in frames[offset - 1 : -1]:  # drop our callers
                    module, lineno, function, _ = frame  # avoid the source code text
                    module = os.path.basename(module)  # lose extension
                    stack = "%s\n%s %s(%d) %s()" % (
                        stack,
                        prefix,
                        module,
                        lineno,
                        function,
                    )
                    prefix = " "
                s = f"{s}:{stack}"
            else:  # unknown location request
                s = f"({loc}?): {s}"
        except Exception:
            pass
    return s


def log_critical(s: object, loc: str = ADCPLogger.critical_loc, alert: str | None = None) -> None:
    """Report string to baselog as a CRITICAL error
    Input:
    s - string
    """
    s = __log_caller_info(s, loc)
    if ADCPLogger.log:
        ADCPLogger.log.critical(s)
    else:
        sys.stderr.write(f"CRITICAL: {s}\n")


log_error_max_count: DefaultDict[str, int] = collections.defaultdict(int)


def log_error(s: str, loc: str = ADCPLogger.error_loc, alert: str | None = None, max_count: int | None = None) -> None:
    """Report string to baselog as an ERROR
    Input:
    s - string
    """
    s = __log_caller_info(s, loc)

    if max_count:
        k = s.split(":")[0]
        log_error_max_count[k] += 1
        if log_error_max_count[k] == max_count:
            s += " (Max message count exceeded)"
        elif log_error_max_count[k] > max_count:
            return

    if ADCPLogger.log:
        ADCPLogger.log.error(s)
    else:
        sys.stderr.write(f"ERROR: {s}\n")


log_warning_max_count: DefaultDict[str, int] = collections.defaultdict(int)


def log_warning(
    s: object, loc: str = ADCPLogger.warning_loc, alert: str | None = None, max_count: int | None = None
) -> None:
    """Report string to baselog as a WARNING
    Input:
    s - string to be logged
    alert - string indicating the class of alert this warning should be assigned to
    max_count - maximum number of times this warning should be issued.
                if a positive value, the count is indexed by the module name and line number
                if a negative value, the count is indexed by the module name, line number andwarning string
    """
    s = __log_caller_info(s, loc)

    if max_count:
        k = s.split(":")[0]
        if max_count < 0:
            k = f"{k}:{s}"
        log_warning_max_count[k] += 1
        if log_warning_max_count[k] == abs(max_count):
            s += " (Max message count exceeded)"
        elif log_warning_max_count[k] > abs(max_count):
            return

    if ADCPLogger.log:
        ADCPLogger.log.warning(s)
    else:
        sys.stderr.write(f"WARNING: {s}\n")


log_info_max_count: DefaultDict[str, int] = collections.defaultdict(int)


def log_info(s: object, loc: str = ADCPLogger.info_loc, alert: str | None = None, max_count: int | None = None) -> None:
    """Report string to baselog as an ERROR
    Input:
    s - string
    """
    if not ADCPLogger.info_enabled:
        return
    s = __log_caller_info(s, loc)

    if max_count:
        k = s.split(":")[0]
        log_info_max_count[k] += 1
        if log_info_max_count[k] == max_count:
            s += " (Max message count exceeded)"
        elif log_info_max_count[k] > max_count:
            return

    if ADCPLogger.log:
        ADCPLogger.log.info(s)
    else:
        sys.stderr.write(f"INFO: {s}\n")


log_debug_max_count: DefaultDict[str, int] = collections.defaultdict(int)


def log_debug(
    s: object, loc: str | None = ADCPLogger.debug_loc, alert: str | None = None, max_count: int | None = None
) -> None:
    """Report string to baselog as DEBUG info
    Input:
    s - string
    """
    if not ADCPLogger.debug_enabled:
        return
    s = __log_caller_info(s, loc)

    if max_count:
        k = s.split(":")[0]
        log_debug_max_count[k] += 1
        if log_debug_max_count[k] == max_count:
            s += " (Max message count exceeded)"
        elif log_debug_max_count[k] > max_count:
            return

    if ADCPLogger.log:
        ADCPLogger.log.debug(s)
    else:
        sys.stderr.write(f"DEBUG: {s}\n")
