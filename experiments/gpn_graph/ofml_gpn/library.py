"""gpn_smc.library -- ProtocolLibrarySpec, moved verbatim from
u15_protocol_backend.py, per K2_DESIGN.md K2a.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolLibrarySpec:
    """Operation-type-library hyperparameters (precommitted defaults).

    Every type shares one amplitude (0.70).  Layer-1 types (A, D) use
    ``layer1_lengthscale`` (1.0) on the fixed warped input; layer-2 types
    (B, C) use ``layer2_lengthscale`` (0.50) on the concatenated
    layer-1-output induced points; the terminal type R uses
    ``terminal_lengthscale`` (0.50) on the concatenated penultimate outputs
    of all four protocols.
    """

    amplitude: float = 0.70
    layer1_lengthscale: float = 1.0
    layer2_lengthscale: float = 0.50
    terminal_lengthscale: float = 0.50
