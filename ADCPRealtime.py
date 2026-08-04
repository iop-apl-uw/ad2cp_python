## IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
## GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
## OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""ADCPRealtine.py - Utilities related to ADCP realtime data processing."""

import numpy as np
import numpy.typing as npt
from scipy import linalg

import ADCPConfig
import ADCPFiles
import ADCPUtils


def _heading_tilt_matrices(
    hh: float, pp: float, rr: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Builds the heading and tilt rotation matrices for one ensemble.

    Args:
        hh: Heading, in radians, offset by -90 degrees (see ``ad2cp_matlab_files.m``).
        pp: Pitch, in radians.
        rr: Roll, in radians.

    Returns:
        A tuple ``(H, P)`` of the 3x3 heading and tilt matrices; ``H @ P`` is the
        instrument-to-ENU (``xyz2enu``) transformation matrix.
    """
    # Make heading matrix
    # H = [cos(hh) sin(hh) 0; -sin(hh) cos(hh) 0; 0 0 1];
    H = np.array([[np.cos(hh), np.sin(hh), 0.0], [-np.sin(hh), np.cos(hh), 0.0], [0.0, 0.0, 1.0]])

    # Make tilt matrix
    # P = [cos(pp) -sin(pp)*sin(rr) -cos(rr)*sin(pp);...
    #   0             cos(rr)          -sin(rr);  ...
    #   sin(pp) sin(rr)*cos(pp)  cos(pp)*cos(rr)];
    P = np.array(
        [
            [np.cos(pp), -np.sin(pp) * np.sin(rr), -np.cos(rr) * np.sin(pp)],
            [0.0, np.cos(rr), -np.sin(rr)],
            [np.sin(pp), np.sin(rr) * np.cos(pp), np.cos(pp) * np.cos(rr)],
        ]
    )
    return H, P


def TransformToInstrument(
    adcp_realtime: ADCPFiles.ADCPRealtimeData, glider: ADCPFiles.SGData, param: ADCPConfig.Params
) -> None:
    """Transforms realtime ADCP velocities between the instrument and Earth (ENU) frames.

    Updates ``adcp_realtime`` in place, setting ``Ux``/``Uy``/``Uz`` to the
    instrument-frame velocities and ``U``/``V``/``W`` to the Earth-frame (ENU)
    velocities, according to ``adcp_realtime.coordinateSystem``:

    - ``coordinateSystem == 0``: the raw ``U``/``V``/``W`` are already ENU: the
      inverse heading/tilt rotation is applied to compute ``Ux``/``Uy``/``Uz``.
    - ``coordinateSystem == 1``: the raw ``U``/``V``/``W`` are already instrument
      frame (copied to ``Ux``/``Uy``/``Uz``); the forward heading/tilt rotation is
      applied to compute Earth-frame ``U``/``V``/``W``, overwriting the raw
      values. If ``param.use_glider_compass``, the glider's own heading/pitch/roll
      (interpolated onto ``adcp_realtime.time``) are used for the rotation instead
      of the ADCP's own, matching ``ad2cp_matlab_files.m``.

    Args:
        adcp_realtime: Realtime ADCP data, read/written in place.
        glider: Glider engineering data, used for the ``coordinateSystem == 1``
            glider-compass override.
        param: Inverse processing parameters; ``param.use_glider_compass`` gates
            the glider-compass override for ``coordinateSystem == 1``.

    Raises:
        ValueError: If ``adcp_realtime.coordinateSystem`` is an unsupported or unknown value.
    """
    if adcp_realtime.coordinateSystem == 0:
        # ENU coordinateSystem (0)

        # B.VelENU(:,:,1)=adcp_realtime.U;
        # B.VelENU(:,:,2)=adcp_realtime.V;
        # B.VelENU(:,:,3)=adcp_realtime.W;
        VelENU = np.array([adcp_realtime[ii] for ii in ("U", "V", "W")])

        # B.VelXYZ = B.VelENU*NaN;
        VelXYZ = VelENU * np.nan

        for nn in range(np.shape(adcp_realtime.pitch)[0]):
            # heading, pitch and roll are the angles output in the data in degrees
            hh = np.pi * (adcp_realtime.heading[nn] - 90.0) / 180.0
            pp = np.pi * adcp_realtime.pitch[nn] / 180.0
            rr = np.pi * adcp_realtime.roll[nn] / 180.0

            # ENU = [B.VelENU(:,nn,1)'; B.VelENU(:,nn,2)'; B.VelENU(:,nn,3)'];
            ENU = np.array([VelENU[ii, :, nn] for ii in range(3)])

            H, P = _heading_tilt_matrices(hh, pp, rr)

            # xyz2enu = H*P;
            # xyz = inv(xyz2enu)*ENU;
            xyz2enu = H @ P
            xyz = linalg.inv(xyz2enu) @ ENU
            for ii in range(3):
                VelXYZ[ii, :, nn] = xyz[ii, :]

        # adcp_realtime.Ux = B.VelXYZ(:,:,1);
        # adcp_realtime.Uy = B.VelXYZ(:,:,2);
        # adcp_realtime.Uz = B.VelXYZ(:,:,3);
        adcp_realtime.Ux = VelXYZ[0, :, :]
        adcp_realtime.Uy = VelXYZ[1, :, :]
        adcp_realtime.Uz = VelXYZ[2, :, :]

    elif adcp_realtime.coordinateSystem == 1:
        # Real-time data is already in instrument (XYZ) frame.
        if param.use_glider_compass:
            # Use the glider's heading, pitch and roll instead of the ADCP's own.
            adcp_realtime.heading = ADCPUtils.course_interp(
                glider.time, glider.eng_head + glider.magnetic_variation, adcp_realtime.time
            )
            adcp_realtime.pitch = ADCPUtils.interp1d(glider.time, glider.eng_pitchAng, adcp_realtime.time)
            adcp_realtime.roll = ADCPUtils.interp1d(glider.time, glider.eng_rollAng, adcp_realtime.time)

        adcp_realtime.Ux = adcp_realtime.U.copy()
        adcp_realtime.Uy = adcp_realtime.V.copy()
        adcp_realtime.Uz = adcp_realtime.W.copy()

        # Forward-transform instrument frame -> Earth (ENU) frame, overwriting the raw
        # U/V/W (which downstream cleaning/inversion consume as Earth-frame velocities).
        VelXYZ = np.array([adcp_realtime[ii] for ii in ("Ux", "Uy", "Uz")])
        VelENU = VelXYZ * np.nan
        for nn in range(np.shape(adcp_realtime.pitch)[0]):
            # heading, pitch and roll are the angles output in the data in degrees
            hh = np.pi * (adcp_realtime.heading[nn] - 90.0) / 180.0
            pp = np.pi * adcp_realtime.pitch[nn] / 180.0
            rr = np.pi * adcp_realtime.roll[nn] / 180.0

            xyz = np.array([VelXYZ[ii, :, nn] for ii in range(3)])

            H, P = _heading_tilt_matrices(hh, pp, rr)
            xyz2enu = H @ P
            enu = xyz2enu @ xyz
            for ii in range(3):
                VelENU[ii, :, nn] = enu[ii, :]

        adcp_realtime.U = VelENU[0, :, :]
        adcp_realtime.V = VelENU[1, :, :]
        adcp_realtime.W = VelENU[2, :, :]

    elif adcp_realtime.coordinateSystem == 2:
        raise ValueError("Beam coordinateSystem not supported")
    else:
        raise ValueError(f"Unknown coordinateSystem {adcp_realtime.coordinateSystem}")

