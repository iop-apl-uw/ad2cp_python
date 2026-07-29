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

import pathlib

import numpy as np
import pytest

import ADCPConfig


def _write_yaml(tmp_path: pathlib.Path, name: str, text: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_process_config_file_no_file_returns_defaults():
    params, weights = ADCPConfig.ProcessConfigFile("")
    assert params is not None
    assert weights is not None
    assert params.dz == 5.0
    assert weights.W_MEAS == 1


def test_process_config_file_missing_file_logged_returns_defaults(tmp_path: pathlib.Path):
    missing = tmp_path / "does_not_exist.yml"
    params, weights = ADCPConfig.ProcessConfigFile(missing)
    # Failure to open/parse is logged and swallowed; cfg_dict falls back to {}
    # for both keys, so this still returns valid, all-default objects.
    assert params is not None
    assert weights is not None


def test_process_config_file_malformed_yaml_logged_returns_defaults(tmp_path: pathlib.Path):
    bad = _write_yaml(tmp_path, "bad.yml", "params: [1, 2\n")
    params, weights = ADCPConfig.ProcessConfigFile(bad)
    assert params is not None
    assert weights is not None


def test_process_config_file_unknown_param_key_raises_validation_error(tmp_path: pathlib.Path):
    cfg = _write_yaml(tmp_path, "cfg.yml", "params:\n  bogus_key: 1\n")
    params, weights = ADCPConfig.ProcessConfigFile(cfg)
    assert (params, weights) == (None, None)


def test_process_config_file_weights_null_returns_none_none(tmp_path: pathlib.Path):
    cfg = _write_yaml(tmp_path, "cfg.yml", "weights:\nparams:\n  dz: 10.0\n")
    params, weights = ADCPConfig.ProcessConfigFile(cfg)
    assert (params, weights) == (None, None)


def test_process_config_file_valid_overrides_applied(tmp_path: pathlib.Path):
    cfg = _write_yaml(
        tmp_path,
        "cfg.yml",
        "params:\n  dz: 10.0\n  sg_id: 686\nweights:\n  W_MEAS: 3\n",
    )
    params, weights = ADCPConfig.ProcessConfigFile(cfg)
    assert params is not None
    assert weights is not None
    assert params.dz == 10.0
    assert params.sg_id == 686
    assert weights.W_MEAS == 3


def test_params_rejects_non_ndarray_time_limits():
    # pydantic's arbitrary_types_allowed still enforces isinstance(value, np.ndarray)
    # before __post_init__ runs, so a raw list is rejected here rather than reaching
    # __post_init__'s np.array() conversion/except branch (which is consequently
    # unreachable via public construction - see plan notes).
    with pytest.raises(Exception, match="instance of ndarray"):
        ADCPConfig.Params(time_limits=[[1, 2], [3]])  # ty: ignore[invalid-argument-type]


def test_params_post_init_leaves_valid_ndarray_time_limits_unchanged():
    arr = np.array([1, 2, 3], dtype=np.int32)
    params = ADCPConfig.Params(time_limits=arr)
    assert params.time_limits is arr


def test_load_var_meta_missing_file_logged_returns_empty(tmp_path: pathlib.Path):
    missing = tmp_path / "does_not_exist.yml"
    assert ADCPConfig.LoadVarMeta(missing) == {}


def test_load_var_meta_non_dict_entry_skipped(tmp_path: pathlib.Path):
    vm = _write_yaml(tmp_path, "var_meta.yml", "some_var: not_a_dict\n")
    assert ADCPConfig.LoadVarMeta(vm) == {}


def test_load_var_meta_validation_error_entry_skipped(tmp_path: pathlib.Path):
    vm = _write_yaml(
        tmp_path,
        "var_meta.yml",
        "some_var:\n  nc_varname: foo\n  nc_dimensions: ['x']\n  nc_type: f\n  decimal_pts: 2\n",
    )
    # Missing required nc_attribs -> ValidationError, entry skipped.
    assert ADCPConfig.LoadVarMeta(vm) == {}


def test_load_var_meta_valid_entry_loaded(tmp_path: pathlib.Path):
    vm = _write_yaml(
        tmp_path,
        "var_meta.yml",
        (
            "some_var:\n"
            "  nc_varname: foo\n"
            "  nc_dimensions: ['x']\n"
            "  nc_attribs:\n"
            "    FillValue: -999\n"
            "    description: A var\n"
            "    units: m\n"
            "    coverage_content_type: physicalMeasurement\n"
            "  nc_type: f\n"
            "  decimal_pts: 2\n"
        ),
    )
    result = ADCPConfig.LoadVarMeta(vm)
    assert set(result) == {"some_var"}
    assert result["some_var"].nc_varname == "foo"


def test_load_global_meta_no_local_override_loads_package_default():
    global_meta = ADCPConfig.LoadGlobalMeta(None)
    assert "global_attributes" in global_meta


def test_load_global_meta_missing_local_override_logged(tmp_path: pathlib.Path):
    missing = tmp_path / "does_not_exist.yml"
    global_meta = ADCPConfig.LoadGlobalMeta(missing)
    # Package default still loaded; local override failure is logged and swallowed.
    assert "global_attributes" in global_meta


def test_load_global_meta_merges_local_override(tmp_path: pathlib.Path):
    local = _write_yaml(
        tmp_path,
        "global_meta_local.yml",
        'global_attributes:\n  institution: "Test Institution"\n',
    )
    global_meta = ADCPConfig.LoadGlobalMeta(local)
    assert global_meta["global_attributes"]["institution"] == "Test Institution"
    # Package default keys are still present (merged, not replaced).
    assert "platform" in global_meta["global_attributes"]


def test_merge_dict_recurses_nested_dicts():
    a = {"a": {"x": 1}}
    b = {"a": {"y": 2}}
    assert ADCPConfig.MergeDict(a, b) == {"a": {"x": 1, "y": 2}}


def test_merge_dict_same_leaf_value_is_noop():
    a = {"a": 1}
    b = {"a": 1}
    assert ADCPConfig.MergeDict(a, b) == {"a": 1}


def test_merge_dict_new_key_added():
    a = {"a": 1}
    b = {"b": 2}
    assert ADCPConfig.MergeDict(a, b) == {"a": 1, "b": 2}


def test_merge_dict_allow_override_replaces_conflicting_leaf():
    a = {"a": 1}
    b = {"a": 2}
    assert ADCPConfig.MergeDict(a, b, allow_override=True) == {"a": 2}


def test_merge_dict_conflict_raises_without_override():
    a = {"a": 1}
    b = {"a": 2}
    with pytest.raises(Exception, match="Conflict at a"):
        ADCPConfig.MergeDict(a, b)
