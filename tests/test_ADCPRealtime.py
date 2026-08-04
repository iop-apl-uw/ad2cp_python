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
import ADCPFiles
import ADCPRealtime
import ADCPUtils

# Real dive with realtime ADCP data reported in the instrument (XYZ) frame
# (coordinateSystem == 1) - sg197_RIOT_Aug26, dive 7.
XYZ_FRAME_NCF = pathlib.Path("testdata/xyz_frame/p1970007.nc")


def test_beam_coordinate_system_raises():
    adcp_realtime = ADCPFiles.ADCPRealtimeData(coordinateSystem=2)
    with pytest.raises(ValueError, match="Beam coordinateSystem not supported"):
        ADCPRealtime.TransformToInstrument(adcp_realtime, ADCPFiles.SGData(), ADCPConfig.Params())


def test_unknown_coordinate_system_raises():
    adcp_realtime = ADCPFiles.ADCPRealtimeData(coordinateSystem=3)
    with pytest.raises(ValueError, match="Unknown coordinateSystem"):
        ADCPRealtime.TransformToInstrument(adcp_realtime, ADCPFiles.SGData(), ADCPConfig.Params())


def test_xyz_coordinate_system_identity_rotation():
    # With heading=90, pitch=0, roll=0 the heading/tilt rotation matrices are both
    # the identity, so the instrument -> ENU transform should be a no-op: Ux/Uy/Uz
    # are the raw instrument-frame values, and U/V/W (now Earth frame) equal them.
    adcp_realtime = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=1,
        pitch=np.array([0.0]),
        roll=np.array([0.0]),
        heading=np.array([90.0]),
        U=np.array([[1.0], [2.0]]),
        V=np.array([[3.0], [4.0]]),
        W=np.array([[5.0], [6.0]]),
    )
    raw_U, raw_V, raw_W = adcp_realtime.U.copy(), adcp_realtime.V.copy(), adcp_realtime.W.copy()
    ADCPRealtime.TransformToInstrument(adcp_realtime, ADCPFiles.SGData(), ADCPConfig.Params(use_glider_compass=False))
    np.testing.assert_array_equal(adcp_realtime.Ux, raw_U)
    np.testing.assert_array_equal(adcp_realtime.Uy, raw_V)
    np.testing.assert_array_equal(adcp_realtime.Uz, raw_W)
    np.testing.assert_allclose(adcp_realtime.U, raw_U)
    np.testing.assert_allclose(adcp_realtime.V, raw_V)
    np.testing.assert_allclose(adcp_realtime.W, raw_W)


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
    ADCPRealtime.TransformToInstrument(adcp_realtime, ADCPFiles.SGData(), ADCPConfig.Params())
    np.testing.assert_allclose(adcp_realtime.Ux, adcp_realtime.U)
    np.testing.assert_allclose(adcp_realtime.Uy, adcp_realtime.V)
    np.testing.assert_allclose(adcp_realtime.Uz, adcp_realtime.W)


def test_xyz_then_enu_round_trips_to_original_instrument_frame():
    # Forward-transforming instrument-frame data to Earth frame (coordinateSystem=1),
    # then inverse-transforming that Earth-frame result back (coordinateSystem=0),
    # should reproduce the original instrument-frame velocities.
    param = ADCPConfig.Params(use_glider_compass=False)
    glider = ADCPFiles.SGData()
    pitch = np.array([12.0, -6.0])
    roll = np.array([-8.0, 4.0])
    heading = np.array([37.0, 210.0])

    forward = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=1,
        pitch=pitch,
        roll=roll,
        heading=heading,
        U=np.array([[1.0, -2.0], [2.0, 0.5], [-3.0, 4.0]]),
        V=np.array([[3.0, 1.0], [4.0, -1.5], [0.5, 2.0]]),
        W=np.array([[5.0, 0.0], [6.0, 3.0], [1.0, -1.0]]),
    )
    ADCPRealtime.TransformToInstrument(forward, glider, param)

    backward = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=0,
        pitch=pitch,
        roll=roll,
        heading=heading,
        U=forward.U,
        V=forward.V,
        W=forward.W,
    )
    ADCPRealtime.TransformToInstrument(backward, glider, param)

    np.testing.assert_allclose(backward.Ux, forward.Ux, atol=1e-12)
    np.testing.assert_allclose(backward.Uy, forward.Uy, atol=1e-12)
    np.testing.assert_allclose(backward.Uz, forward.Uz, atol=1e-12)


def test_xyz_coordinate_system_uses_glider_compass_when_enabled():
    # The ADCP's own heading/pitch/roll are non-trivial (would rotate the data), but
    # the glider's compass reports an identity rotation (heading=90, pitch=0, roll=0)
    # over a time window bracketing the single ensemble. With use_glider_compass=True
    # the glider's compass must win, so U/V/W (now Earth frame) equal the raw input.
    adcp_realtime = ADCPFiles.ADCPRealtimeData(
        coordinateSystem=1,
        time=np.array([100.0]),
        pitch=np.array([45.0]),
        roll=np.array([45.0]),
        heading=np.array([0.0]),
        U=np.array([[1.0], [2.0]]),
        V=np.array([[3.0], [4.0]]),
        W=np.array([[5.0], [6.0]]),
    )
    raw_U, raw_V, raw_W = adcp_realtime.U.copy(), adcp_realtime.V.copy(), adcp_realtime.W.copy()
    glider = ADCPFiles.SGData(
        time=np.array([0.0, 200.0]),
        eng_head=np.array([90.0, 90.0]),
        eng_pitchAng=np.array([0.0, 0.0]),
        eng_rollAng=np.array([0.0, 0.0]),
        magnetic_variation=0.0,
    )
    ADCPRealtime.TransformToInstrument(adcp_realtime, glider, ADCPConfig.Params(use_glider_compass=True))

    np.testing.assert_allclose(adcp_realtime.heading, [90.0])
    np.testing.assert_allclose(adcp_realtime.pitch, [0.0])
    np.testing.assert_allclose(adcp_realtime.roll, [0.0])
    np.testing.assert_allclose(adcp_realtime.U, raw_U)
    np.testing.assert_allclose(adcp_realtime.V, raw_V)
    np.testing.assert_allclose(adcp_realtime.W, raw_W)


def test_xyz_coordinate_system_real_dive_transforms_to_earth_frame():
    # Regression test for the coordinateSystem==1 fix, using a real dive (sg197_RIOT_Aug26,
    # dive 7) whose realtime ADCP data was reported in the instrument (XYZ) frame - before
    # the fix, U/V/W were left as these raw, un-rotated instrument-frame values and fed
    # straight into CleanADCP/Inverse as if they were already Earth-frame (ENU).
    param = ADCPConfig.Params()
    ds = ADCPUtils.open_netcdf_file(XYZ_FRAME_NCF, mode="r")
    assert ds is not None
    glider, gps, adcp_realtime = ADCPFiles.ADCPReadSGNCF(ds, XYZ_FRAME_NCF, param)
    ds.close()
    assert adcp_realtime.coordinateSystem == 1

    raw_U, raw_V, raw_W = adcp_realtime.U.copy(), adcp_realtime.V.copy(), adcp_realtime.W.copy()
    ADCPRealtime.TransformToInstrument(adcp_realtime, glider, param)

    # Instrument frame Ux/Uy/Uz are the raw, un-rotated velocities.
    np.testing.assert_array_equal(adcp_realtime.Ux, raw_U)
    np.testing.assert_array_equal(adcp_realtime.Uy, raw_V)
    np.testing.assert_array_equal(adcp_realtime.Uz, raw_W)

    # Earth-frame U/V/W must actually have been rotated away from the raw instrument
    # values (the bug this fixture reproduces: they were previously left untouched).
    assert not np.allclose(adcp_realtime.U, raw_U, equal_nan=True)
    assert not np.allclose(adcp_realtime.V, raw_V, equal_nan=True)
    assert not np.allclose(adcp_realtime.W, raw_W, equal_nan=True)

    # The heading/tilt rotation is orthogonal, so per-ensemble vector magnitude is
    # preserved going from instrument frame to Earth frame - a real-data correctness
    # check that doesn't require hand-deriving the expected rotated values.
    raw_norm = np.sqrt(raw_U**2 + raw_V**2 + raw_W**2)
    enu_norm = np.sqrt(adcp_realtime.U**2 + adcp_realtime.V**2 + adcp_realtime.W**2)
    np.testing.assert_allclose(enu_norm, raw_norm, atol=1e-10)
