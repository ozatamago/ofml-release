"""Operation-composing OFML model interfaces."""

from .base import OFMLModel, OperationModel
from .bayesian_terminal_only import (
    BayesianTerminalOnlyConfig,
    BayesianTerminalOnlyOFML,
)
from .closed_form_gp import ClosedFormGPOperationModel
from .linear_composed import LinearComposedConfig, LinearComposedOFML
from .terminal_only import TerminalOnlyConfig, TerminalOnlyOFML

try:  # botorch is optional; only FullyBayesianGPOperationModel needs it
    from .bayesian_gp import FullyBayesianGPOperationModel
except Exception:  # pragma: no cover - env without botorch
    FullyBayesianGPOperationModel = None
from .operation_models import (
    DKLOperationModel,
    EnsembleOperationModel,
    GPOperationModel,
    LinearOperationModel,
    MCDOperationModel,
    MLPOperationModel,
    OperationMLPConfig,
    UnknownOperationModel,
    read_operation_examples,
)

__all__ = [
    "ClosedFormGPOperationModel",
    "BayesianTerminalOnlyOFML",
    "BayesianTerminalOnlyConfig",
    "LinearComposedOFML",
    "LinearComposedConfig",
    "TerminalOnlyOFML",
    "TerminalOnlyConfig",
    "DKLOperationModel",
    "EnsembleOperationModel",
    "FullyBayesianGPOperationModel",
    "GPOperationModel",
    "LinearOperationModel",
    "MCDOperationModel",
    "MLPOperationModel",
    "OFMLModel",
    "OperationModel",
    "OperationMLPConfig",
    "UnknownOperationModel",
    "read_operation_examples",
]
