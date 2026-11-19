"""gpn_smc -- shared SMC backend package extracted from
``u15_protocol_backend.py`` (K2_DESIGN.md K2a; plan v2 K1..K7). Re-exports
every public name the original monolithic module defined (or imported and
re-exposed), so ``from gpn_smc import *`` reproduces
``u15_protocol_backend``'s public surface exactly; see
``ucb/noisy_ucb_experiments/experiments/u15_protocol_backend.py`` (now a
thin shim) and
``ucb/noisy_ucb_experiments/experiments/u15_protocol_backend_pre_k2a_2026-09-02.py.bak``
(the frozen pre-K2a reference copy).

Sets ``JAX_ENABLE_X64``/``XLA_PYTHON_CLIENT_PREALLOCATE`` before any ``jax``
import is triggered by this package's submodules (matching the original
module's own module-scope ``os.environ.setdefault`` calls, which ran before
its ``import jax``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from .constants import (
    ADAPTIVE_BETA_INITIAL,
    ADAPTIVE_BETA_MAX,
    ADAPTIVE_BETA_MIN,
    ADAPTIVE_BETA_TARGET_ACCEPTANCE,
    GRID_SIZE_L2N,
    GRID_SIZE_L2W,
    GRID_SIZE_M10,
    GRID_SIZE_M12,
    GRID_SIZE_PER_PROTOCOL,
    LOG2PI,
    N_CANDIDATES,
    N_CANDIDATES_L2N,
    N_CANDIDATES_L2W,
    N_CANDIDATES_M9,
    N_CANDIDATES_M10,
    N_CANDIDATES_M12,
    N_L1_TYPES_L2N,
    N_L1_TYPES_L2W,
    N_L1_TYPES_M12,
    N_L2_TYPES_L2N,
    N_L2_TYPES_L2W,
    N_L2_TYPES_M9,
    N_L2_TYPES_M10,
    N_MAX_U15,
    N_PCN_BLOCKS,
    N_PROTOCOLS,
    N_PROTOCOLS_L2N,
    N_PROTOCOLS_L2W,
    N_PROTOCOLS_M9,
    N_PROTOCOLS_M10,
    N_PROTOCOLS_M12,
    N_SIMPLE_PROTOCOLS_M12,
    N_TARGET_PROTOCOLS_M12,
    PCN_SEGMENT_LENGTH,
    PROTOCOL_DEPTHS_M12,
    PROTOCOL_LAYER1_TYPE,
    PROTOCOL_LAYER1_TYPE_M9,
    PROTOCOL_LAYER1_TYPE_M10,
    PROTOCOL_LAYER1_TYPE_L2N,
    PROTOCOL_LAYER1_TYPE_L2W,
    PROTOCOL_LAYER2_TYPE,
    PROTOCOL_LAYER2_TYPE_M9,
    PROTOCOL_LAYER2_TYPE_M10,
    PROTOCOL_LAYER2_TYPE_L2N,
    PROTOCOL_LAYER2_TYPE_L2W,
    RAMP_W_LIB_B,
    SEGMENT_BETA_MIN,
)
from .numerics import (
    EXTENDED_QUANTILE_LEVELS,
    c_schedule,
    compute_chunk_size,
    derive_stream_seed,
    normalize_logweights,
    rbf_covariance,
    stable_cholesky,
    systematic_resample,
)
from .pcn import PCN_MODES, PcnBetaState, SegmentRejuvenationInfo, adapt_pcn_beta
from .library import ProtocolLibrarySpec
from .pushforward import (
    pushforward_h,
    pushforward_shared_h,
    pushforward_shared_h_battery,
    pushforward_shared_h_general,
    pushforward_shared_h_general_nl1,
    pushforward_shared_h_lib_s,
    pushforward_shared_h_m12,
    shared_pcn_block_bounds,
    shared_pcn_block_bounds_m12,
)
from .terminal_rb import build_terminal_rb_functions
from .backends import (
    IndependentProtocolGpnBackend,
    SharedProtocolGpnBackend,
    split_flat_observations_by_protocol,
)
from .assimilate import (
    assimilate_batches_u15,
    assimilate_u15,
    padded_observations_u15,
)
# BENCH9 additive (2026-11-01b): generic library-graph language and its SMC backends.
from .graph_library import (
    GRAPH_LIBRARY_BUILDERS,
    GraphLibrary,
    GraphNode,
    GraphType,
    draw_graph_teacher,
    get_graph_library,
    rbf_kernel_nd,
)
from .graph_backends import GraphFactorisedIndependentGpnBackend, GraphSharedGpnBackend
from .coverage_quantiles import (
    COVERAGE_LEVELS,
    indep_quantiles,
    shared_quantiles,
)

__all__ = [
    "EXTENDED_QUANTILE_LEVELS",
    "c_schedule",
    "derive_stream_seed",
    "normalize_logweights",
    "rbf_covariance",
    "stable_cholesky",
    "systematic_resample",
    "compute_chunk_size",
    "pushforward_h",
    "LOG2PI",
    "N_MAX_U15",
    "N_PROTOCOLS",
    "GRID_SIZE_PER_PROTOCOL",
    "N_CANDIDATES",
    "PROTOCOL_LAYER1_TYPE",
    "PROTOCOL_LAYER2_TYPE",
    "N_L2_TYPES_M9",
    "N_PROTOCOLS_M9",
    "N_CANDIDATES_M9",
    "PROTOCOL_LAYER1_TYPE_M9",
    "PROTOCOL_LAYER2_TYPE_M9",
    "N_L2_TYPES_M10",
    "N_PROTOCOLS_M10",
    "N_CANDIDATES_M10",
    "GRID_SIZE_M10",
    "PROTOCOL_LAYER1_TYPE_M10",
    "PROTOCOL_LAYER2_TYPE_M10",
    "N_L1_TYPES_L2W",
    "N_L2_TYPES_L2W",
    "N_PROTOCOLS_L2W",
    "N_CANDIDATES_L2W",
    "GRID_SIZE_L2W",
    "PROTOCOL_LAYER1_TYPE_L2W",
    "PROTOCOL_LAYER2_TYPE_L2W",
    "N_L1_TYPES_L2N",
    "N_L2_TYPES_L2N",
    "N_PROTOCOLS_L2N",
    "N_CANDIDATES_L2N",
    "GRID_SIZE_L2N",
    "PROTOCOL_LAYER1_TYPE_L2N",
    "PROTOCOL_LAYER2_TYPE_L2N",
    "N_L1_TYPES_M12",
    "N_SIMPLE_PROTOCOLS_M12",
    "N_TARGET_PROTOCOLS_M12",
    "N_PROTOCOLS_M12",
    "N_CANDIDATES_M12",
    "GRID_SIZE_M12",
    "PROTOCOL_DEPTHS_M12",
    "RAMP_W_LIB_B",
    "PCN_MODES",
    "N_PCN_BLOCKS",
    "ADAPTIVE_BETA_INITIAL",
    "ADAPTIVE_BETA_TARGET_ACCEPTANCE",
    "ADAPTIVE_BETA_MIN",
    "ADAPTIVE_BETA_MAX",
    "PCN_SEGMENT_LENGTH",
    "SEGMENT_BETA_MIN",
    "adapt_pcn_beta",
    "SegmentRejuvenationInfo",
    "PcnBetaState",
    "ProtocolLibrarySpec",
    "pushforward_shared_h",
    "pushforward_shared_h_general",
    "pushforward_shared_h_general_nl1",
    "pushforward_shared_h_m12",
    "pushforward_shared_h_lib_s",
    "pushforward_shared_h_battery",
    "shared_pcn_block_bounds_m12",
    "shared_pcn_block_bounds",
    "build_terminal_rb_functions",
    "SharedProtocolGpnBackend",
    "split_flat_observations_by_protocol",
    "IndependentProtocolGpnBackend",
    "padded_observations_u15",
    "assimilate_u15",
    "assimilate_batches_u15",
    "GRAPH_LIBRARY_BUILDERS",
    "GraphLibrary",
    "GraphNode",
    "GraphType",
    "draw_graph_teacher",
    "get_graph_library",
    "rbf_kernel_nd",
    "GraphSharedGpnBackend",
    "GraphFactorisedIndependentGpnBackend",
    "COVERAGE_LEVELS",
    "indep_quantiles",
    "shared_quantiles",
]
