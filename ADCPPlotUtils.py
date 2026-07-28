#! /usr/bin/env python
# -*- python-fmt -*-
## Copyright (c) 2023, 2024, 2025  University of Washington.
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

"""ADCPPlotUtils.py - Utility functions for plotting."""

import cmocean
import numpy as np


def cmocean_to_plotly(cmapname: str, pl_entries: int) -> list[list[float | str]]:
    """Converts a cmocean colormap into a plotly colorscale.

    Args:
        cmapname: Name of the cmocean colormap (e.g. ``"balance"``); falls back to
            ``"thermal"`` if not recognized.
        pl_entries: Number of colorscale stops to generate.

    Returns:
        Plotly colorscale: a list of ``[position, "rgb(r, g, b)"]`` entries, with
        ``position`` in ``[0, 1]``.
    """
    # cmocean.cm injects its colormap names into the module namespace dynamically
    # (locals().update(...)), so they aren't visible to static attribute access.
    names = [
        "thermal",
        "haline",
        "solar",
        "ice",
        "gray",
        "oxy",
        "deep",
        "dense",
        "algae",
        "matter",
        "turbid",
        "speed",
        "amp",
        "tempo",
        "phase",
        "balance",
        "delta",
        "curl",
    ]
    maps = {name: getattr(cmocean.cm, name) for name in names}

    cmap = maps.get(cmapname, maps["thermal"])

    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale
