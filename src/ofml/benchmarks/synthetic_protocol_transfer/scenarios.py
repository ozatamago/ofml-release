"""Scenario definitions for synthetic protocol-transfer benchmarks."""

from __future__ import annotations

import dataclasses
from enum import Enum


class ScenarioName(str, Enum):
    """Known synthetic protocol-transfer scenarios."""

    S0_SUCCESS = "s0_success"
    S1_UNKNOWN_OPERATION = "s1_unknown_operation"
    S2_SEMANTICS_SHIFT = "s2_semantics_shift"
    S3_LOCAL_INPUT_EXTRAPOLATION = "s3_local_input_extrapolation"
    S4_MISSING_PROBE = "s4_missing_probe"
    S5_LIPSCHITZ_INSTABILITY = "s5_lipschitz_instability"
    S6_MISSPECIFICATION = "s6_misspecification"
    S7_THREE_BODY_INTERACTION = "s7_three_body_interaction"
    S8_DOSE_COUNT_SATURATION = "s8_dose_count_saturation"
    NONLINEAR_S0_SUCCESS = "nonlinear_s0_success"
    NONLINEAR_S1_UNKNOWN_OPERATION = "nonlinear_s1_unknown_operation"
    NONLINEAR_S2_SEMANTICS_SHIFT = "nonlinear_s2_semantics_shift"
    NONLINEAR_S3_LOCAL_INPUT_EXTRAPOLATION = "nonlinear_s3_local_input_extrapolation"
    NONLINEAR_S4_MISSING_PROBE = "nonlinear_s4_missing_probe"
    NONLINEAR_S5_LIPSCHITZ_INSTABILITY = "nonlinear_s5_lipschitz_instability"
    NONLINEAR_S6_MISSPECIFICATION = "nonlinear_s6_misspecification"
    NONLINEAR_S7_THREE_BODY_INTERACTION = "nonlinear_s7_three_body_interaction"
    BO_SMOOTH_S0_SUCCESS = "bo_smooth_s0_success"
    BO_SMOOTH_S1_UNKNOWN_OPERATION = "bo_smooth_s1_unknown_operation"
    BO_SMOOTH_S2_SEMANTICS_SHIFT = "bo_smooth_s2_semantics_shift"
    BO_SMOOTH_S3_LOCAL_INPUT_EXTRAPOLATION = "bo_smooth_s3_local_input_extrapolation"
    BO_SMOOTH_S4_MISSING_PROBE = "bo_smooth_s4_missing_probe"
    BO_SMOOTH_S5_LIPSCHITZ_INSTABILITY = "bo_smooth_s5_lipschitz_instability"
    BO_SMOOTH_S6_MISSPECIFICATION = "bo_smooth_s6_misspecification"
    BO_SMOOTH_S7_THREE_BODY_INTERACTION = "bo_smooth_s7_three_body_interaction"
    BO_MULTIMODAL_READOUT_S0_SUCCESS = "bo_multimodal_readout_s0_success"
    BO_MULTIMODAL_READOUT_S1_UNKNOWN_OPERATION = "bo_multimodal_readout_s1_unknown_operation"
    BO_MULTIMODAL_READOUT_S2_SEMANTICS_SHIFT = "bo_multimodal_readout_s2_semantics_shift"
    BO_MULTIMODAL_READOUT_S3_LOCAL_INPUT_EXTRAPOLATION = "bo_multimodal_readout_s3_local_input_extrapolation"
    BO_MULTIMODAL_READOUT_S4_MISSING_PROBE = "bo_multimodal_readout_s4_missing_probe"
    BO_MULTIMODAL_READOUT_S5_LIPSCHITZ_INSTABILITY = "bo_multimodal_readout_s5_lipschitz_instability"
    BO_MULTIMODAL_READOUT_S6_MISSPECIFICATION = "bo_multimodal_readout_s6_misspecification"
    BO_MULTIMODAL_READOUT_S7_THREE_BODY_INTERACTION = "bo_multimodal_readout_s7_three_body_interaction"
    BO_OPERATION_MULTIMODAL_S0_SUCCESS = "bo_operation_multimodal_s0_success"
    BO_OPERATION_MULTIMODAL_S1_UNKNOWN_OPERATION = "bo_operation_multimodal_s1_unknown_operation"
    BO_OPERATION_MULTIMODAL_S2_SEMANTICS_SHIFT = "bo_operation_multimodal_s2_semantics_shift"
    BO_OPERATION_MULTIMODAL_S3_LOCAL_INPUT_EXTRAPOLATION = "bo_operation_multimodal_s3_local_input_extrapolation"
    BO_OPERATION_MULTIMODAL_S4_MISSING_PROBE = "bo_operation_multimodal_s4_missing_probe"
    BO_OPERATION_MULTIMODAL_S5_LIPSCHITZ_INSTABILITY = "bo_operation_multimodal_s5_lipschitz_instability"
    BO_OPERATION_MULTIMODAL_S6_MISSPECIFICATION = "bo_operation_multimodal_s6_misspecification"
    BO_OPERATION_MULTIMODAL_S7_THREE_BODY_INTERACTION = "bo_operation_multimodal_s7_three_body_interaction"
    NOISE_HETEROSCEDASTIC_S0_SUCCESS = "noise_heteroscedastic_s0_success"
    NOISE_HETEROSCEDASTIC_S1_UNKNOWN_OPERATION = "noise_heteroscedastic_s1_unknown_operation"
    NOISE_HETEROSCEDASTIC_S2_SEMANTICS_SHIFT = "noise_heteroscedastic_s2_semantics_shift"
    NOISE_HETEROSCEDASTIC_S3_LOCAL_INPUT_EXTRAPOLATION = "noise_heteroscedastic_s3_local_input_extrapolation"
    NOISE_HETEROSCEDASTIC_S4_MISSING_PROBE = "noise_heteroscedastic_s4_missing_probe"
    NOISE_HETEROSCEDASTIC_S5_LIPSCHITZ_INSTABILITY = "noise_heteroscedastic_s5_lipschitz_instability"
    NOISE_HETEROSCEDASTIC_S6_MISSPECIFICATION = "noise_heteroscedastic_s6_misspecification"
    NOISE_HETEROSCEDASTIC_S7_THREE_BODY_INTERACTION = "noise_heteroscedastic_s7_three_body_interaction"
    NOISE_HEAVY_TAILED_S0_SUCCESS = "noise_heavy_tailed_s0_success"
    NOISE_HEAVY_TAILED_S1_UNKNOWN_OPERATION = "noise_heavy_tailed_s1_unknown_operation"
    NOISE_HEAVY_TAILED_S2_SEMANTICS_SHIFT = "noise_heavy_tailed_s2_semantics_shift"
    NOISE_HEAVY_TAILED_S3_LOCAL_INPUT_EXTRAPOLATION = "noise_heavy_tailed_s3_local_input_extrapolation"
    NOISE_HEAVY_TAILED_S4_MISSING_PROBE = "noise_heavy_tailed_s4_missing_probe"
    NOISE_HEAVY_TAILED_S5_LIPSCHITZ_INSTABILITY = "noise_heavy_tailed_s5_lipschitz_instability"
    NOISE_HEAVY_TAILED_S6_MISSPECIFICATION = "noise_heavy_tailed_s6_misspecification"
    NOISE_HEAVY_TAILED_S7_THREE_BODY_INTERACTION = "noise_heavy_tailed_s7_three_body_interaction"
    PROTOCOL_INPUT_MISSINGNESS = "protocol_input_missingness"
    PROTOCOL_LONG_CHAIN = "protocol_long_chain"
    PROTOCOL_BRANCHING_REPEATED = "protocol_branching_repeated"
    NONCOMMUTATIVE_LINEAR = "noncommutative_linear"
    NONCOMMUTATIVE_NONLINEAR = "noncommutative_nonlinear"


class ProtocolRole(str, Enum):
    """How a protocol should be assigned to dataset splits."""

    BOTH = "both"
    SOURCE = "source"
    TARGET = "target"


@dataclasses.dataclass(frozen=True)
class ScenarioDefinition:
    """Human-readable scenario metadata."""

    name: ScenarioName
    description: str


SCENARIOS: dict[ScenarioName, ScenarioDefinition] = {
    ScenarioName.S0_SUCCESS: ScenarioDefinition(
        name=ScenarioName.S0_SUCCESS,
        description="All target protocol operations are represented in training.",
    ),
    ScenarioName.S1_UNKNOWN_OPERATION: ScenarioDefinition(
        name=ScenarioName.S1_UNKNOWN_OPERATION,
        description="The target protocol contains an operation absent from training.",
    ),
    ScenarioName.S2_SEMANTICS_SHIFT: ScenarioDefinition(
        name=ScenarioName.S2_SEMANTICS_SHIFT,
        description="The target protocol reuses known operation names with shifted semantics.",
    ),
    ScenarioName.S3_LOCAL_INPUT_EXTRAPOLATION: ScenarioDefinition(
        name=ScenarioName.S3_LOCAL_INPUT_EXTRAPOLATION,
        description="The target protocol composes known operations so local operation inputs leave training support.",
    ),
    ScenarioName.S4_MISSING_PROBE: ScenarioDefinition(
        name=ScenarioName.S4_MISSING_PROBE,
        description="All target operations appear in training, but probes are insufficient to identify operation behavior.",
    ),
    ScenarioName.S5_LIPSCHITZ_INSTABILITY: ScenarioDefinition(
        name=ScenarioName.S5_LIPSCHITZ_INSTABILITY,
        description="A long high-Lipschitz operation chain amplifies small operation-level errors.",
    ),
    ScenarioName.S6_MISSPECIFICATION: ScenarioDefinition(
        name=ScenarioName.S6_MISSPECIFICATION,
        description="The assumed operation factorization is intentionally misspecified.",
    ),
    ScenarioName.S7_THREE_BODY_INTERACTION: ScenarioDefinition(
        name=ScenarioName.S7_THREE_BODY_INTERACTION,
        description="The target depends on a higher-order interaction not visible in pairs.",
    ),
    ScenarioName.S8_DOSE_COUNT_SATURATION: ScenarioDefinition(
        name=ScenarioName.S8_DOSE_COUNT_SATURATION,
        description=(
            "The target readout depends on the NUMBER of times a dose operation is "
            "repeated (a saturating count-graded effect, #12), which is invisible to "
            "operation-presence descriptors and requires an operation-count "
            "(multiset) descriptor to resolve -- one descriptor level above s7's "
            "presence gate."
        ),
    ),
    ScenarioName.NONLINEAR_S0_SUCCESS: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S0_SUCCESS,
        description="Smooth nonlinear operations with fully represented target protocol operations.",
    ),
    ScenarioName.NONLINEAR_S1_UNKNOWN_OPERATION: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S1_UNKNOWN_OPERATION,
        description="Smooth nonlinear operations with an unseen target operation.",
    ),
    ScenarioName.NONLINEAR_S2_SEMANTICS_SHIFT: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S2_SEMANTICS_SHIFT,
        description="Smooth nonlinear operations with shifted target semantics.",
    ),
    ScenarioName.NONLINEAR_S3_LOCAL_INPUT_EXTRAPOLATION: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S3_LOCAL_INPUT_EXTRAPOLATION,
        description="Smooth nonlinear operations under local input extrapolation.",
    ),
    ScenarioName.NONLINEAR_S4_MISSING_PROBE: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S4_MISSING_PROBE,
        description="Smooth nonlinear operations with insufficient operation-identifying probes.",
    ),
    ScenarioName.NONLINEAR_S5_LIPSCHITZ_INSTABILITY: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S5_LIPSCHITZ_INSTABILITY,
        description="Smooth nonlinear operations with amplified chain errors.",
    ),
    ScenarioName.NONLINEAR_S6_MISSPECIFICATION: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S6_MISSPECIFICATION,
        description="Smooth but high-frequency operations outside the low-capacity model class.",
    ),
    ScenarioName.NONLINEAR_S7_THREE_BODY_INTERACTION: ScenarioDefinition(
        name=ScenarioName.NONLINEAR_S7_THREE_BODY_INTERACTION,
        description="Smooth nonlinear operations with a target-only three-body interaction.",
    ),
    ScenarioName.PROTOCOL_INPUT_MISSINGNESS: ScenarioDefinition(
        name=ScenarioName.PROTOCOL_INPUT_MISSINGNESS,
        description="Operation-level supervision is input-dependently missing in selected regions.",
    ),
    ScenarioName.PROTOCOL_LONG_CHAIN: ScenarioDefinition(
        name=ScenarioName.PROTOCOL_LONG_CHAIN,
        description="Repeated long-chain protocols quantify error propagation as chain length grows.",
    ),
    ScenarioName.PROTOCOL_BRANCHING_REPEATED: ScenarioDefinition(
        name=ScenarioName.PROTOCOL_BRANCHING_REPEATED,
        description="Repeated operations are reused under different state distributions.",
    ),
    ScenarioName.NONCOMMUTATIVE_LINEAR: ScenarioDefinition(
        name=ScenarioName.NONCOMMUTATIVE_LINEAR,
        description=(
            "Affine non-commuting operations trained on single and canonical-order pair "
            "probes; targets are a sweep of operation orderings (e.g. B->C->A->O) so "
            "compositional generalization to unseen orders can be measured."
        ),
    ),
    ScenarioName.NONCOMMUTATIVE_NONLINEAR: ScenarioDefinition(
        name=ScenarioName.NONCOMMUTATIVE_NONLINEAR,
        description=(
            "Smooth nonlinear non-commuting operations with the same single/pair training "
            "and ordering-sweep targets, contrasting local MLP operation models against "
            "globally extrapolating affine ones under operation reordering."
        ),
    ),
}

_BO_SCENARIO_DESCRIPTIONS = {
    "s0_success": "all target protocol operations represented in source protocols",
    "s1_unknown_operation": "an unseen target operation",
    "s2_semantics_shift": "a shifted target semantic variant",
    "s3_local_input_extrapolation": "local input extrapolation",
    "s4_missing_probe": "insufficient operation-identifying probes",
    "s5_lipschitz_instability": "long-chain Lipschitz error amplification",
    "s6_misspecification": "high-frequency misspecification",
    "s7_three_body_interaction": "a target-only three-body interaction",
}

for _prefix, _label in (
    ("bo_smooth", "BO smooth compositional functions"),
    ("bo_multimodal_readout", "BO multimodal readout functions"),
    ("bo_operation_multimodal", "BO operation-level multimodal functions"),
    ("noise_heteroscedastic", "BO smooth functions with heteroscedastic operation/readout noise"),
    ("noise_heavy_tailed", "BO smooth functions with heavy-tailed operation/readout noise"),
):
    for _suffix, _description in _BO_SCENARIO_DESCRIPTIONS.items():
        _scenario_name = ScenarioName(f"{_prefix}_{_suffix}")
        SCENARIOS[_scenario_name] = ScenarioDefinition(
            name=_scenario_name,
            description=f"{_label} with {_description}.",
        )


def parse_scenario(value: str | ScenarioName) -> ScenarioName:
    """Parse a scenario value from config."""

    if isinstance(value, ScenarioName):
        return value
    try:
        return ScenarioName(str(value))
    except ValueError as error:
        allowed = ", ".join(scenario.value for scenario in ScenarioName)
        raise ValueError(f"Unknown scenario '{value}'. Expected one of: {allowed}.") from error


def parse_protocol_role(value: str | ProtocolRole) -> ProtocolRole:
    """Parse a protocol split role from config."""

    if isinstance(value, ProtocolRole):
        return value
    try:
        return ProtocolRole(str(value))
    except ValueError as error:
        allowed = ", ".join(role.value for role in ProtocolRole)
        raise ValueError(f"Unknown protocol role '{value}'. Expected one of: {allowed}.") from error
