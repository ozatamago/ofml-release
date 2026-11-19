"""gpn_smc.constants -- module-level constants moved verbatim from
u15_protocol_backend.py (LOG2PI; the N_MAX_U15/N_PROTOCOLS/GRID_SIZE_PER_
PROTOCOL/N_CANDIDATES M8 default and the M9/M10/M11/M12 library-size
families; RAMP_W_LIB_B; the non-PCN_MODES pCN constants N_PCN_BLOCKS,
ADAPTIVE_BETA_*, PCN_SEGMENT_LENGTH, SEGMENT_BETA_MIN -- PCN_MODES itself
lives in gpn_smc.pcn alongside the pCN state machinery it enumerates). Every
right-hand-side expression below is character-identical to the source
(verified by ast.dump comparison in the K2a build script); only the
explanatory banner comments that separated these blocks in the monolithic
module were not reproduced here (they remain in
u15_protocol_backend_pre_k2a_2026-09-02.py.bak and the original file).
Per K2_DESIGN.md K2a.
"""

from __future__ import annotations

import math

import numpy as np


LOG2PI = math.log(2.0 * math.pi)

# M10f additive: raised 40 -> 100 to let production runs reach
# total_observations=100 (bucketed padding via n_max_buckets on the
# backends handles the intermediate sizes; see backends.py's
# n_max_for()/_bucket_kernels).
N_MAX_U15 = 100

N_PROTOCOLS = 4

GRID_SIZE_PER_PROTOCOL = 64

N_CANDIDATES = N_PROTOCOLS * GRID_SIZE_PER_PROTOCOL  # 256

PROTOCOL_LAYER1_TYPE = ("A", "A", "D", "D")

PROTOCOL_LAYER2_TYPE = ("B", "C", "B", "C")

N_L2_TYPES_M9 = 4

N_PROTOCOLS_M9 = 2 * N_L2_TYPES_M9  # 8

N_CANDIDATES_M9 = N_PROTOCOLS_M9 * GRID_SIZE_PER_PROTOCOL  # 512

PROTOCOL_LAYER1_TYPE_M9 = ("A", "D", "A", "D", "A", "D", "A", "D")

PROTOCOL_LAYER2_TYPE_M9 = ("B", "B", "C", "C", "G", "G", "H", "H")

N_L2_TYPES_M10 = 8

N_PROTOCOLS_M10 = 2 * N_L2_TYPES_M10  # 16

N_CANDIDATES_M10 = N_PROTOCOLS_M10 * GRID_SIZE_PER_PROTOCOL  # 1024

GRID_SIZE_M10 = 2 * GRID_SIZE_PER_PROTOCOL * (1 + N_L2_TYPES_M10)  # 1152

PROTOCOL_LAYER1_TYPE_M10 = tuple(["A"] * 8 + ["B"] * 8)

PROTOCOL_LAYER2_TYPE_M10 = tuple([f"C{j}" for j in range(1, 9)] * 2)

N_L1_TYPES_L2W = 4

N_L2_TYPES_L2W = 2

N_PROTOCOLS_L2W = N_L1_TYPES_L2W * N_L2_TYPES_L2W  # 8

N_CANDIDATES_L2W = N_PROTOCOLS_L2W * GRID_SIZE_PER_PROTOCOL  # 512

GRID_SIZE_L2W = N_L1_TYPES_L2W * GRID_SIZE_PER_PROTOCOL * (1 + N_L2_TYPES_L2W)  # 768

PROTOCOL_LAYER1_TYPE_L2W = tuple(
    f"A{i}" for i in range(N_L1_TYPES_L2W) for _ in range(N_L2_TYPES_L2W)
)

PROTOCOL_LAYER2_TYPE_L2W = tuple(
    f"C{j}" for _ in range(N_L1_TYPES_L2W) for j in range(N_L2_TYPES_L2W)
)

N_L1_TYPES_L2N = 2

N_L2_TYPES_L2N = 4

N_PROTOCOLS_L2N = N_L1_TYPES_L2N * N_L2_TYPES_L2N  # 8

N_CANDIDATES_L2N = N_PROTOCOLS_L2N * GRID_SIZE_PER_PROTOCOL  # 512

GRID_SIZE_L2N = N_L1_TYPES_L2N * GRID_SIZE_PER_PROTOCOL * (1 + N_L2_TYPES_L2N)  # 640

PROTOCOL_LAYER1_TYPE_L2N = tuple(
    f"A{i}" for i in range(N_L1_TYPES_L2N) for _ in range(N_L2_TYPES_L2N)
)

PROTOCOL_LAYER2_TYPE_L2N = tuple(
    f"C{j}" for _ in range(N_L1_TYPES_L2N) for j in range(N_L2_TYPES_L2N)
)

N_L1_TYPES_M12 = 4

N_SIMPLE_PROTOCOLS_M12 = N_L1_TYPES_M12  # 4 (protocols 0-3)

N_TARGET_PROTOCOLS_M12 = N_L1_TYPES_M12  # 4 (protocols 4-7)

N_PROTOCOLS_M12 = N_SIMPLE_PROTOCOLS_M12 + N_TARGET_PROTOCOLS_M12  # 8

N_CANDIDATES_M12 = N_PROTOCOLS_M12 * GRID_SIZE_PER_PROTOCOL  # 512

GRID_SIZE_M12 = 3 * N_L1_TYPES_M12 * GRID_SIZE_PER_PROTOCOL  # 768

PROTOCOL_DEPTHS_M12 = (2,) * N_SIMPLE_PROTOCOLS_M12 + (4,) * N_TARGET_PROTOCOLS_M12

RAMP_W_LIB_B = np.linspace(-2.0, 2.0, GRID_SIZE_PER_PROTOCOL)

N_PCN_BLOCKS = 4

ADAPTIVE_BETA_INITIAL = 0.30

ADAPTIVE_BETA_TARGET_ACCEPTANCE = 0.23

ADAPTIVE_BETA_MIN = 0.01

ADAPTIVE_BETA_MAX = 0.50

PCN_SEGMENT_LENGTH = 8

SEGMENT_BETA_MIN = 0.002
