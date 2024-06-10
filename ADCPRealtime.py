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
ADCPRealtine.py - Utilities related to ADCP realtime data processing
"""

import numpy as np
from scipy import linalg

import ADCPFiles


def TransformToInstrument(adcp_realtime: ADCPFiles.ADCPRealtimeData) -> None:
    """
    Performs the inverse transformation to get the velocities in the instruments frame
    """
    # B.VelENU(:,:,1)=adcp_realtime.U;
    # B.VelENU(:,:,2)=adcp_realtime.V;
    # B.VelENU(:,:,3)=adcp_realtime.W;
    # VelENU = np.array([dsi.variables[f"ad2cp_vel{ii}"][:] for ii in ("X", "Y", "Z")])
    VelENU = np.array([adcp_realtime[ii] for ii in ("U", "V", "W")])

    # B.Pitch = adcp_realtime.pitch;
    # B.Roll = adcp_realtime.roll;
    # B.Heading = adcp_realtime.heading;
    # B.VelXYZ = B.VelENU*NaN;
    VelXYZ = VelENU * np.nan

    # for nn = 1:length(B.Pitch)
    for nn in range(np.shape(adcp_realtime.pitch)[0]):
        #   % heading, pitch and roll are the angles output in the data in degrees
        #   hh = pi*(B.Heading(nn)-90)/180';
        #   pp = pi*B.Pitch(nn)/180';
        #   rr = pi*B.Roll(nn)/180';
        hh = np.pi * (adcp_realtime.heading[nn] - 90.0) / 180.0
        pp = np.pi * adcp_realtime.pitch[nn] / 180.0
        rr = np.pi * adcp_realtime.roll[nn] / 180.0

        #   ENU = [B.VelENU(:,nn,1)';...
        #     B.VelENU(:,nn,2)';...
        #     B.VelENU(:,nn,3)'];
        ENU = np.array([VelENU[ii, nn, :] for ii in range(3)])

        #   % Make heading matrix
        #   H = [cos(hh) sin(hh) 0; -sin(hh) cos(hh) 0; 0 0 1];
        H = np.array([[np.cos(hh), np.sin(hh), 0.0], [-np.sin(hh), np.cos(hh), 0.0], [0.0, 0.0, 1.0]])

        #   % Make tilt matrix
        #   P = [cos(pp) -sin(pp)*sin(rr) -cos(rr)*sin(pp);...
        #     0             cos(rr)          -sin(rr);  ...
        #     sin(pp) sin(rr)*cos(pp)  cos(pp)*cos(rr)];
        P = np.array(
            [
                [np.cos(pp), -np.sin(pp) * np.sin(rr), -np.cos(rr) * np.sin(pp)],
                [0.0, np.cos(rr), -np.sin(rr)],
                [np.sin(pp), np.sin(rr) * np.cos(pp), np.cos(pp) * np.cos(rr)],
            ]
        )

        #   % Make resulting transformation matrix
        #   xyz2enu = H*P;
        #   xyz = inv(xyz2enu)*ENU;
        #   B.VelXYZ(:,nn,1) = xyz(1,:)';
        #   B.VelXYZ(:,nn,2) = xyz(2,:)';
        #   B.VelXYZ(:,nn,3) = xyz(3,:)';
        # end
        xyz2enu = H @ P
        xyz = linalg.inv(xyz2enu) @ ENU
        for ii in range(3):
            # VelXYZ[ii, :, nn] = xyz[ii, :]
            # FORTRAN vs C
            VelXYZ[ii, nn, :] = xyz[ii, :]

        # adcp_realtime.Ux = B.VelXYZ(:,:,1);
        # adcp_realtime.Uy = B.VelXYZ(:,:,2);
        # adcp_realtime.Uz = B.VelXYZ(:,:,3);
        # Ux, Uy, Uz = [VelXYZ[ii, :, :] for ii in range(3)]
        adcp_realtime.Ux = VelXYZ[0, :, :]
        adcp_realtime.Uy = VelXYZ[1, :, :]
        adcp_realtime.Uz = VelXYZ[2, :, :]
