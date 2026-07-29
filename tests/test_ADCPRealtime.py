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

import numpy as np
import pytest

import ADCPFiles
import ADCPRealtime


def test_beam_coordinate_system_raises():
    adcp_realtime = ADCPFiles.ADCPRealtimeData(coordinateSystem=2)
    with pytest.raises(ValueError, match="Beam coordinateSystem not supported"):
        ADCPRealtime.TransformToInstrument(adcp_realtime)


def test_unknown_coordinate_system_raises():
    adcp_realtime = ADCPFiles.ADCPRealtimeData(coordinateSystem=3)
    with pytest.raises(ValueError, match="Unknown coordinateSystem"):
        ADCPRealtime.TransformToInstrument(adcp_realtime)


def test_xyz_coordinate_system_passthrough():
    adcp_realtime = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=1,
        U=np.array([1.0, 2.0]),
        V=np.array([3.0, 4.0]),
        W=np.array([5.0, 6.0]),
    )
    ADCPRealtime.TransformToInstrument(adcp_realtime)
    np.testing.assert_array_equal(adcp_realtime.Ux, adcp_realtime.U)
    np.testing.assert_array_equal(adcp_realtime.Uy, adcp_realtime.V)
    np.testing.assert_array_equal(adcp_realtime.Uz, adcp_realtime.W)


def test_enu_coordinate_system_identity_rotation():
    # With heading=90, pitch=0, roll=0 the heading/tilt rotation matrices are both
    # the identity, so the ENU -> instrument transform should be a no-op.
    adcp_realtime = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=0,
        pitch=np.array([0.0]),
        roll=np.array([0.0]),
        heading=np.array([90.0]),
        U=np.array([[1.0], [2.0]]),
        V=np.array([[3.0], [4.0]]),
        W=np.array([[5.0], [6.0]]),
    )
    ADCPRealtime.TransformToInstrument(adcp_realtime)
    np.testing.assert_allclose(adcp_realtime.Ux, adcp_realtime.U)
    np.testing.assert_allclose(adcp_realtime.Uy, adcp_realtime.V)
    np.testing.assert_allclose(adcp_realtime.Uz, adcp_realtime.W)
