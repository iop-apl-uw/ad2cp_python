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

import h5py
import numpy as np
import pytest

import CompareToMat

_COMPLEX_FIELD_DTYPE = np.dtype([("real", "f8"), ("imag", "f8")])


def test_read_matlab_nested_group_and_2d_swapaxes(tmp_path: pathlib.Path):
    path = tmp_path / "test.mat"
    with h5py.File(path, "w") as f:
        grp = f.create_group("adcp")
        grp.create_dataset("Z", data=np.arange(12).reshape(3, 4).astype(np.float64))
        grp.create_dataset("Z0", data=np.array([1.0, 2.0, 3.0]))

    result = CompareToMat.read_matlab(path)
    assert set(result) == {"adcp"}
    # 2D arrays are swapaxes'd (FORTRAN -> C order).
    np.testing.assert_array_equal(result["adcp"]["Z"], np.arange(12).reshape(3, 4).T)
    # 1D arrays pass through unchanged.
    np.testing.assert_array_equal(result["adcp"]["Z0"], np.array([1.0, 2.0, 3.0]))


def test_read_matlab_object_reference_cell_resolves_to_path(tmp_path: pathlib.Path):
    path = tmp_path / "test.mat"
    with h5py.File(path, "w") as f:
        target = f.create_dataset("target_data", data=np.array([1.0, 2.0, 3.0]))
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        refs = f.create_dataset("cell_refs", (2,), dtype=ref_dtype)
        refs[0] = target.ref
        refs[1] = target.ref

    result = CompareToMat.read_matlab(path)
    assert result["cell_refs"] == ["/target_data", "/target_data"]


def test_read_matlab_string_cell_excluded(tmp_path: pathlib.Path):
    path = tmp_path / "test.mat"
    with h5py.File(path, "w") as f:
        vlen_dtype = h5py.special_dtype(vlen=bytes)
        strs = f.create_dataset("str_cell", (2,), dtype=vlen_dtype)
        strs[0] = b"hello"
        strs[1] = b"world"

    result = CompareToMat.read_matlab(path)
    # String/bytes cell entries are excluded (False), not resolved as references.
    assert result["str_cell"] == [False, False]


def _build_synthetic_files(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Builds a matching pair of (python_file, mat_file) HDF5 files.

    Covers every variable pair CompareToMat.main() compares, so it can run end
    to end without KeyErrors (main() has no per-pair error handling - a
    missing key crashes the whole comparison loop).

    Most pairs get identical trivial placeholder data (hits the "all_close"
    branch); a handful are deliberately varied to exercise main()'s other
    conditional branches (shape mismatch, complex real/imag field conversion).
    """
    default = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    py_data: dict[tuple[str, str], np.ndarray] = {}
    mat_data: dict[tuple[str, str], np.ndarray] = {}
    for _f_plot, py_grp, py_name, mat_grp, mat_name in CompareToMat.COMPARISON_VARS:
        py_data.setdefault((py_grp, py_name), default)
        mat_data.setdefault((mat_grp, mat_name), default)

    # Shape mismatch, but broadcastable (mat side has size-1 trailing dim).
    mat_data[("glider", "ctd_depth")] = np.array([2.0])
    # Shape mismatch, NOT broadcastable.
    mat_data[("glider", "Wmod")] = np.zeros((3, 4))
    # Complex real/imag field conversion: python side complex, matlab side a
    # compound (real, imag) record array - matches how MATLAB v7.3 actually
    # stores complex data in HDF5.
    py_data[("glider", "UV1")] = default.astype(np.complex128) + 1j * default
    mat_uv1 = np.zeros(default.shape[0], dtype=_COMPLEX_FIELD_DTYPE)
    mat_uv1["real"] = default
    mat_uv1["imag"] = default * 0.5
    mat_data[("glider", "UV1")] = mat_uv1

    python_path = tmp_path / "python_output.hdf5"
    with h5py.File(python_path, "w") as f:
        for (grp, name), arr in py_data.items():
            f.require_group(grp).create_dataset(name, data=arr)

    mat_path = tmp_path / "matlab_output.mat"
    with h5py.File(mat_path, "w") as f:
        for (grp, name), arr in mat_data.items():
            f.require_group(grp).create_dataset(name, data=arr)

    return python_path, mat_path


def test_main_runs_full_comparison_without_raising(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    python_path, mat_path = _build_synthetic_files(tmp_path)
    CompareToMat.main([str(mat_path), str(python_path)])
    captured = capsys.readouterr()
    # Most pairs are identical placeholders (all_close); the deliberately-varied
    # ones should surface as NOT_CLOSE or shape-mismatch output.
    assert "all_close" in captured.out
    assert "NOT_CLOSE" in captured.out
