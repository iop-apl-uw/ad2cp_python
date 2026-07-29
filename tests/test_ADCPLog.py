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

import argparse
import logging
import pathlib
import traceback
from collections.abc import Callable

import pytest

import ADCPLog


def _opts(adcp_log: str | None = None, debug: bool = False, verbose: bool = False) -> argparse.Namespace:
    return argparse.Namespace(adcp_log=adcp_log, debug=debug, verbose=verbose)


def test_adcp_logger_console_only_default_level():
    ADCPLog.ADCPLogger(_opts())
    assert ADCPLog.ADCPLogger.is_initialized
    assert ADCPLog.ADCPLogger.log is not None
    handlers = ADCPLog.ADCPLogger.log.handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.WARNING


def test_adcp_logger_verbose_sets_info_enabled():
    ADCPLog.ADCPLogger(_opts(verbose=True))
    assert ADCPLog.ADCPLogger.info_enabled
    assert not ADCPLog.ADCPLogger.debug_enabled
    assert ADCPLog.ADCPLogger.log is not None
    assert ADCPLog.ADCPLogger.log.handlers[0].level == logging.INFO


def test_adcp_logger_debug_sets_debug_and_info_enabled():
    ADCPLog.ADCPLogger(_opts(debug=True))
    assert ADCPLog.ADCPLogger.debug_enabled
    assert ADCPLog.ADCPLogger.info_enabled
    assert ADCPLog.ADCPLogger.log is not None
    assert ADCPLog.ADCPLogger.log.handlers[0].level == logging.DEBUG


def test_adcp_logger_none_opts_defaults_warning():
    ADCPLog.ADCPLogger(None)
    assert ADCPLog.ADCPLogger.log is not None
    assert ADCPLog.ADCPLogger.log.handlers[0].level == logging.WARNING


def test_adcp_logger_file_handler_created(tmp_path: pathlib.Path):
    log_file = tmp_path / "adcp.log"
    ADCPLog.ADCPLogger(_opts(adcp_log=str(log_file)))
    assert ADCPLog.ADCPLogger.log is not None
    handlers = ADCPLog.ADCPLogger.log.handlers
    assert any(isinstance(h, logging.FileHandler) for h in handlers)
    ADCPLog.log_error("boom")
    for h in handlers:
        h.flush()
    assert "boom" in log_file.read_text()


def test_adcp_logger_second_construction_is_noop():
    ADCPLog.ADCPLogger(_opts(verbose=True))
    ADCPLog.ADCPLogger(_opts(debug=True))
    assert ADCPLog.ADCPLogger.info_enabled
    assert not ADCPLog.ADCPLogger.debug_enabled


def test_get_logger_returns_shared_logger():
    logger = ADCPLog.ADCPLogger(_opts())
    result = logger.getLogger()
    assert result is ADCPLog.ADCPLogger.log


def test_get_logger_uninitialized_defensive_branch():
    # Contrived state: log already exists from a prior construction, but
    # is_initialized has been forced back to False - exercises getLogger()'s
    # defensive "not is_initialized" pass-branch, which real code never hits
    # since __init__ always sets both together.
    logger = ADCPLog.ADCPLogger(_opts())
    ADCPLog.ADCPLogger.is_initialized = False
    result = logger.getLogger()
    assert result is ADCPLog.ADCPLogger.log


def _log_caller_info_via_wrapper(s: str, loc: str) -> str:
    # Mimics the single log_XXXX() frame that normally sits between a real
    # caller and __log_caller_info, matching the "__log_caller_info();
    # log_XXXX; <caller>" frame-skipping the "caller"/"stack" offsets assume.
    return ADCPLog.__log_caller_info(s, loc)


def test_log_caller_info_caller_loc_prepends_module_line():
    result = _log_caller_info_via_wrapper("hello", "caller")
    assert result.startswith("test_ADCPLog.py(")
    assert result.endswith("): hello")


def _log_caller_info_via_two_wrappers(s: str, loc: str) -> str:
    # An extra layer versus _log_caller_info_via_wrapper, matching "parent"'s
    # intended use: a utility routine calling a log_XXXX-like function on
    # behalf of its own caller.
    return _log_caller_info_via_wrapper(s, loc)


def test_log_caller_info_parent_loc_points_one_frame_up():
    result = _log_caller_info_via_two_wrappers("hello", "parent")
    assert result.startswith("test_ADCPLog.py(")
    assert result.endswith("): hello")


def test_log_caller_info_exc_loc_appends_traceback_when_exception_active():
    try:
        raise ValueError("boom")
    except ValueError:
        result = ADCPLog.__log_caller_info("oops", "exc")
    assert result.startswith("oops:\n")
    assert "ValueError: boom" in result


def test_log_caller_info_exc_loc_outside_exception_appends_none_placeholder():
    # traceback.format_exc() returns the literal string "NoneType: None\n" (not
    # an empty string) when there's no active exception, so the "if exc:"
    # guard is always true in practice - documenting actual behavior rather
    # than the "no-op" behavior the source comment implies.
    result = ADCPLog.__log_caller_info("oops", "exc")
    assert result == "oops:\nNoneType: None\n"


def test_log_caller_info_stack_loc_includes_full_stack():
    result = _log_caller_info_via_wrapper("hello", "stack")
    assert result.startswith("hello:")
    assert "test_ADCPLog.py" in result


def test_log_caller_info_unknown_loc_wraps_string():
    result = ADCPLog.__log_caller_info("hello", "bogus")
    assert result == "(bogus?): hello"


def test_log_caller_info_none_loc_passthrough():
    assert ADCPLog.__log_caller_info("hello", None) == "hello"


def test_log_caller_info_swallows_introspection_exception(monkeypatch: pytest.MonkeyPatch):
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("introspection broke")

    monkeypatch.setattr(traceback, "extract_stack", _raise)
    result = ADCPLog.__log_caller_info("hello", "caller")
    assert result == "hello"


@pytest.mark.parametrize(
    "log_func,prefix",
    [
        (ADCPLog.log_error, "ERROR"),
        (ADCPLog.log_warning, "WARNING"),
    ],
)
def test_log_uninitialized_writes_to_stderr(
    log_func: Callable[..., None], prefix: str, capsys: pytest.CaptureFixture[str]
):
    log_func("boom")
    captured = capsys.readouterr()
    assert captured.err.startswith(f"{prefix}: ")
    assert captured.err.rstrip().endswith("boom")


def test_log_critical_uninitialized_writes_to_stderr_with_exc_noise(capsys: pytest.CaptureFixture[str]):
    # log_critical's default loc is "exc", and traceback.format_exc() always
    # returns a truthy string even with no active exception, so every
    # log_critical() call made outside an except block gets "NoneType: None"
    # noise appended (see test_log_caller_info_exc_loc_outside_exception_...
    # above). Documenting actual behavior; not fixed as part of this pass -
    # flagged separately since ADCPUtils.check_versions() hits this in
    # production.
    ADCPLog.log_critical("boom")
    captured = capsys.readouterr()
    assert captured.err == "CRITICAL: boom:\nNoneType: None\n\n"


def test_log_info_and_log_debug_noop_when_disabled(capsys: pytest.CaptureFixture[str]):
    ADCPLog.log_info("boom")
    ADCPLog.log_debug("boom")
    captured = capsys.readouterr()
    assert captured.err == ""


@pytest.mark.parametrize("log_func,prefix", [(ADCPLog.log_info, "INFO"), (ADCPLog.log_debug, "DEBUG")])
def test_log_info_and_log_debug_uninitialized_writes_to_stderr(
    log_func: Callable[..., None], prefix: str, capsys: pytest.CaptureFixture[str]
):
    # Contrived: real code never sets *_enabled True without also setting
    # ADCPLogger.log, but forcing this state exercises the stderr-fallback
    # branch that would otherwise be unreachable.
    ADCPLog.ADCPLogger.info_enabled = True
    ADCPLog.ADCPLogger.debug_enabled = True
    log_func("boom")
    captured = capsys.readouterr()
    assert captured.err.startswith(f"{prefix}: ")
    assert captured.err.rstrip().endswith("boom")


def test_log_error_max_count_suppresses_after_threshold(capsys: pytest.CaptureFixture[str]):
    # All 3 calls share one call site (this same source line), so max_count's
    # positive-mode key (call site only, regardless of message) accumulates
    # across iterations: 1st logs normally, 2nd hits the threshold marker,
    # 3rd is fully suppressed.
    for _ in range(3):
        ADCPLog.log_error("boom", max_count=2)
    captured = capsys.readouterr()
    lines = captured.err.rstrip("\n").split("\n")
    assert len(lines) == 2
    assert lines[1].endswith("(Max message count exceeded)")


def test_log_warning_max_count_suppresses_after_threshold(capsys: pytest.CaptureFixture[str]):
    for _ in range(3):
        ADCPLog.log_warning("boom", max_count=1)
    captured = capsys.readouterr()
    lines = captured.err.rstrip("\n").split("\n")
    assert len(lines) == 1
    assert lines[0].endswith("(Max message count exceeded)")


@pytest.mark.parametrize("log_func,prefix", [(ADCPLog.log_info, "INFO"), (ADCPLog.log_debug, "DEBUG")])
def test_log_info_and_log_debug_max_count_suppresses_after_threshold(
    log_func: Callable[..., None], prefix: str, capsys: pytest.CaptureFixture[str]
):
    # Contrived enabled state (see test_log_info_and_log_debug_uninitialized_
    # writes_to_stderr above) so the max_count logic (which runs regardless
    # of whether a real logger is attached) is reachable at all.
    ADCPLog.ADCPLogger.info_enabled = True
    ADCPLog.ADCPLogger.debug_enabled = True
    for _ in range(3):
        log_func("boom", max_count=2)
    captured = capsys.readouterr()
    lines = captured.err.rstrip("\n").split("\n")
    assert len(lines) == 2
    assert lines[1].endswith("(Max message count exceeded)")


def test_log_critical_uses_real_logger_when_initialized(caplog: pytest.LogCaptureFixture):
    ADCPLog.ADCPLogger(_opts())
    with caplog.at_level("CRITICAL"):
        ADCPLog.log_critical("boom", loc=None)
    assert any(r.levelname == "CRITICAL" and "boom" in r.message for r in caplog.records)


def _log_warning_msg(msg: str, max_count: int) -> None:
    ADCPLog.log_warning(msg, max_count=max_count)


def test_log_warning_max_count_positive_counts_by_call_site_not_message():
    _log_warning_msg("first", 5)
    _log_warning_msg("second", 5)
    assert len(ADCPLog.log_warning_max_count) == 1


def test_log_warning_max_count_negative_counts_by_full_message():
    _log_warning_msg("first", -5)
    _log_warning_msg("second", -5)
    assert len(ADCPLog.log_warning_max_count) == 2
