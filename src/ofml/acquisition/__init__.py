"""Acquisition strategies for active learning."""

from .base import AcquisitionResult
from .design import alc_sampling, embed_rows, kcenter_sampling
from .fantasy import FantasyConfig, deterministic_fantasy_sampling, fantasy_sampling
from .loop import ActiveLearningHistory, run_active_learning
from .pool import CandidatePool, build_candidate_pool
from .random import random_sampling
from .tracking import sample_efficiency, simple_regret
from .uncertainty import uncertainty_sampling

__all__ = [
    "AcquisitionResult",
    "ActiveLearningHistory",
    "CandidatePool",
    "FantasyConfig",
    "alc_sampling",
    "build_candidate_pool",
    "deterministic_fantasy_sampling",
    "embed_rows",
    "fantasy_sampling",
    "kcenter_sampling",
    "random_sampling",
    "run_active_learning",
    "sample_efficiency",
    "simple_regret",
    "uncertainty_sampling",
]
