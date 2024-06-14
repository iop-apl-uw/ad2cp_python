# -*- python-fmt -*-

## Copyright (c) 2024  University of Washington.
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

from dataclasses import dataclass, field

import pytest

import ExtendedDataClass


@dataclass
class edc_example(ExtendedDataClass.ExtendedDataClass):
    x: list | None = None
    y: list | None = None


@dataclass
class edc_example2(ExtendedDataClass.ExtendedDataClass):
    w: list = field(default_factory=list)


@pytest.fixture
def edc():
    """Generates a new ExtendedDataClass with two fields populated"""
    zz = edc_example()
    zz.x = [True, True]
    zz.y = [True, True]
    return zz


def test_type_error(edc):
    with pytest.raises(TypeError):
        # The line below should raise a [attr-defined] error from mypy
        edc.z = True


def test_containment(edc):
    assert "x" in edc
    assert "z" not in edc


def test_key_access(edc):
    edc["x"]
    assert edc["x"] == [True, True]
    for _, v in edc.items():
        assert v == [True, True]


def test_key_enumeration(edc):
    # ruff: noqa: SIM118
    assert [ii for ii in edc.keys()] == ["x", "y"]


def test_key_assigment(edc):
    edc["x"] = [True]
    with pytest.raises(TypeError):
        edc["items"] = True


def test_keyerror(edc):
    with pytest.raises(KeyError):
        print(edc["z"])


def test_mutable_default_value():
    assert edc_example2().w is not edc_example2().w
