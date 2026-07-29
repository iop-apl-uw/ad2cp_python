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

import re

import pytest

import ADCPPlotUtils

_RGB_RE = re.compile(r"^rgb\(\d{1,3}, \d{1,3}, \d{1,3}\)$")


@pytest.mark.parametrize("cmapname", ["balance", "thermal", "haline", "curl"])
def test_cmocean_to_plotly_recognized_names(cmapname: str):
    result = ADCPPlotUtils.cmocean_to_plotly(cmapname, 10)
    assert len(result) == 10
    for position, rgb in result:
        assert isinstance(position, float)
        assert isinstance(rgb, str)
        assert 0.0 <= position <= 1.0
        assert _RGB_RE.match(rgb)
    assert result[0][0] == 0.0


def test_cmocean_to_plotly_unrecognized_name_falls_back_to_thermal():
    result = ADCPPlotUtils.cmocean_to_plotly("not_a_real_colormap", 10)
    expected = ADCPPlotUtils.cmocean_to_plotly("thermal", 10)
    assert result == expected
