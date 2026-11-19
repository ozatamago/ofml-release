"""Synthetic protocol-transfer benchmark generator."""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import yaml

from .scenarios import ProtocolRole, ScenarioName, parse_protocol_role, parse_scenario
from .semantic_variants import apply_step as apply_semantic_step
from .semantic_variants import readout as semantic_readout
from .splits import split_rows


@dataclasses.dataclass(frozen=True)
class TraceMissingRule:
    """Rule for omitting operation-level supervision in selected input regions."""

    operations: tuple[str, ...] | None = None
    step_indices: tuple[int, ...] | None = None
    input_1_low: float | None = None
    input_1_high: float | None = None
    input_2_low: float | None = None
    input_2_high: float | None = None

    def __post_init__(self) -> None:
        if self.operations is not None:
            object.__setattr__(self, "operations", tuple(str(operation) for operation in self.operations))
        if self.step_indices is not None:
            object.__setattr__(self, "step_indices", tuple(int(step_index) for step_index in self.step_indices))


@dataclasses.dataclass(frozen=True)
class ProtocolConfig:
    """Configuration for one synthetic protocol."""

    name: str
    steps: tuple[str, ...]
    n_samples: int
    role: ProtocolRole = ProtocolRole.BOTH
    semantic_variant: str = "default"
    input_low: float | None = None
    input_high: float | None = None
    observed_operations: tuple[str, ...] | None = None
    observed_step_indices: tuple[int, ...] | None = None
    missing_trace_rules: tuple[TraceMissingRule, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(str(step) for step in self.steps))
        object.__setattr__(self, "role", parse_protocol_role(self.role))
        object.__setattr__(self, "semantic_variant", str(self.semantic_variant))
        if self.observed_operations is not None:
            object.__setattr__(
                self,
                "observed_operations",
                tuple(str(operation) for operation in self.observed_operations),
            )
        if self.observed_step_indices is not None:
            object.__setattr__(
                self,
                "observed_step_indices",
                tuple(int(step_index) for step_index in self.observed_step_indices),
            )
        object.__setattr__(self, "missing_trace_rules", tuple(self.missing_trace_rules))


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    """Configuration for generated observation noise."""

    kind: str = "gaussian"
    std: float = 0.0
    base_std: float = 0.0
    peak_std: float = 0.0
    center: tuple[float, ...] = (0.8, 0.8)
    bandwidth: float = 0.18
    df: float = 3.0
    distribution: str = "student_t"


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for synthetic benchmark generation."""

    seed: int
    protocols: tuple[ProtocolConfig, ...]
    output_dir: pathlib.Path
    scenario: ScenarioName = ScenarioName.S0_SUCCESS
    input_low: float = 0.0
    input_high: float = 1.0
    noise_std: float = 0.01
    operation_noise: NoiseConfig | None = None
    readout_noise: NoiseConfig | None = None
    train_fraction: float = 0.75


def load_config(path: str | pathlib.Path) -> BenchmarkConfig:
    """Load a benchmark configuration from YAML."""

    config_path = pathlib.Path(path)
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Benchmark config must be a mapping.")
    return parse_config(raw, base_dir=config_path.parent)


def parse_config(raw: Mapping[str, Any], base_dir: pathlib.Path | None = None) -> BenchmarkConfig:
    """Parse a raw mapping into a benchmark configuration."""

    base_dir = base_dir or pathlib.Path.cwd()
    protocols_raw = raw.get("protocols")
    if not isinstance(protocols_raw, Sequence) or isinstance(protocols_raw, str):
        raise ValueError("Config field 'protocols' must be a sequence.")

    protocols = []
    for item in protocols_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each protocol entry must be a mapping.")
        protocols.append(_parse_protocol_config(item, base_dir))

    output_dir = pathlib.Path(str(raw["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    train_fraction = float(raw.get("train_fraction", 0.75))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("Config field 'train_fraction' must be between 0 and 1.")

    return BenchmarkConfig(
        seed=int(raw["seed"]),
        protocols=tuple(protocols),
        output_dir=output_dir,
        scenario=parse_scenario(raw.get("scenario", ScenarioName.S0_SUCCESS.value)),
        input_low=float(raw.get("input_low", 0.0)),
        input_high=float(raw.get("input_high", 1.0)),
        noise_std=float(raw.get("noise_std", 0.01)),
        operation_noise=_parse_noise_config(raw.get("noise", {}), "operation"),
        readout_noise=_parse_noise_config(raw.get("noise", {}), "readout"),
        train_fraction=train_fraction,
    )


def _parse_noise_config(raw: object, key: str) -> NoiseConfig | None:
    if not isinstance(raw, Mapping):
        return None
    section = raw.get(key)
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError(f"Config field 'noise.{key}' must be a mapping.")
    center_raw = section.get("center", (0.8, 0.8))
    if isinstance(center_raw, Sequence) and not isinstance(center_raw, str):
        center = tuple(float(value) for value in center_raw)
    else:
        raise ValueError(f"Config field 'noise.{key}.center' must be a sequence.")
    return NoiseConfig(
        kind=str(section.get("kind", "gaussian")),
        std=float(section.get("std", 0.0)),
        base_std=float(section.get("base_std", section.get("std", 0.0))),
        peak_std=float(section.get("peak_std", section.get("std", 0.0))),
        center=center,
        bandwidth=float(section.get("bandwidth", 0.18)),
        df=float(section.get("df", 3.0)),
        distribution=str(section.get("distribution", "student_t")),
    )


def _parse_protocol_config(raw: Mapping[str, Any], base_dir: pathlib.Path) -> ProtocolConfig:
    role = parse_protocol_role(raw.get("role", ProtocolRole.BOTH.value))
    semantic_variant = str(raw.get("semantic_variant", "default"))
    input_low = float(raw["input_low"]) if "input_low" in raw else None
    input_high = float(raw["input_high"]) if "input_high" in raw else None
    observed_operations = (
        tuple(str(operation) for operation in raw["observed_operations"])
        if "observed_operations" in raw
        else None
    )
    observed_step_indices = (
        tuple(int(step_index) for step_index in raw["observed_step_indices"])
        if "observed_step_indices" in raw
        else None
    )
    missing_trace_rules = _parse_trace_missing_rules(raw.get("missing_trace_rules", ()))

    if "ofplang_yaml" in raw:
        from ofml.protocols.ofplang_adapter import protocol_config_from_yaml

        yaml_path = pathlib.Path(str(raw["ofplang_yaml"]))
        if not yaml_path.is_absolute():
            yaml_path = (base_dir / yaml_path).resolve()
        config = protocol_config_from_yaml(
            yaml_path,
            n_samples=int(raw["n_samples"]),
            name=str(raw["name"]) if "name" in raw else None,
            step_map=raw.get("step_map"),
            exclude_types=raw.get("exclude_types"),
            exclude_ids=raw.get("exclude_ids"),
        )
        return dataclasses.replace(
            config,
            role=role,
            semantic_variant=semantic_variant,
            input_low=input_low,
            input_high=input_high,
            observed_operations=observed_operations,
            observed_step_indices=observed_step_indices,
            missing_trace_rules=missing_trace_rules,
        )

    return ProtocolConfig(
        name=str(raw["name"]),
        steps=tuple(str(step) for step in raw["steps"]),
        n_samples=int(raw["n_samples"]),
        role=role,
        semantic_variant=semantic_variant,
        input_low=input_low,
        input_high=input_high,
        observed_operations=observed_operations,
        observed_step_indices=observed_step_indices,
        missing_trace_rules=missing_trace_rules,
    )


def _parse_trace_missing_rules(raw: object) -> tuple[TraceMissingRule, ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError("Config field 'missing_trace_rules' must be a sequence.")
    rules = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each missing trace rule must be a mapping.")
        operations = (
            tuple(str(operation) for operation in item["operations"])
            if "operations" in item
            else None
        )
        step_indices = (
            tuple(int(step_index) for step_index in item["step_indices"])
            if "step_indices" in item
            else None
        )
        rules.append(
            TraceMissingRule(
                operations=operations,
                step_indices=step_indices,
                input_1_low=float(item["input_1_low"]) if "input_1_low" in item else None,
                input_1_high=float(item["input_1_high"]) if "input_1_high" in item else None,
                input_2_low=float(item["input_2_low"]) if "input_2_low" in item else None,
                input_2_high=float(item["input_2_high"]) if "input_2_high" in item else None,
            )
        )
    return tuple(rules)


def generate_dataset(config: BenchmarkConfig) -> dict[str, list[dict[str, str]]]:
    """Generate train and test rows for the synthetic benchmark."""

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, str]] = []
    operation_examples: list[dict[str, str]] = []

    for protocol in config.protocols:
        input_low = config.input_low if protocol.input_low is None else protocol.input_low
        input_high = config.input_high if protocol.input_high is None else protocol.input_high
        for sample_id in range(protocol.n_samples):
            x = rng.uniform(input_low, input_high, size=2)
            y, trace = _evaluate_protocol_with_trace(
                x,
                protocol.steps,
                protocol_name=protocol.name,
                protocol_role=protocol.role,
                sample_id=sample_id,
                semantic_variant=protocol.semantic_variant,
                rng=rng,
                operation_noise=config.operation_noise,
                readout_noise=config.readout_noise,
                legacy_noise_std=config.noise_std,
            )
            rows.append(
                {
                    "protocol": protocol.name,
                    "protocol_role": protocol.role.value,
                    "semantic_variant": protocol.semantic_variant,
                    "sample_id": str(sample_id),
                    "steps": " ".join(protocol.steps),
                    "x1": _format_float(x[0]),
                    "x2": _format_float(x[1]),
                    "y": _format_float(float(y)),
                }
            )
            operation_examples.extend(_filter_observed_trace(trace, protocol))

    dataset = split_rows(rows, train_fraction=config.train_fraction, rng=rng)
    split_by_key = {
        (row["protocol"], row["sample_id"]): split_name
        for split_name in ("train", "test")
        for row in dataset[split_name]
    }
    dataset["operation_examples"] = [
        dict(example, split=split_by_key[(example["protocol"], example["sample_id"])])
        for example in operation_examples
    ]
    return dataset


def write_dataset(config: BenchmarkConfig, dataset: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    """Write generated dataset splits and metadata to disk."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["protocol", "protocol_role", "semantic_variant", "sample_id", "steps", "x1", "x2", "y"]

    for split_name in ("train", "test"):
        split_path = config.output_dir / f"{split_name}.csv"
        with split_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset[split_name])

    operation_fieldnames = [
        "split",
        "protocol",
        "protocol_role",
        "semantic_variant",
        "sample_id",
        "step_index",
        "operation",
        "input_1",
        "input_2",
        "output_1",
        "output_2",
        "output_dim",
    ]
    with (config.output_dir / "operation_examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=operation_fieldnames)
        writer.writeheader()
        writer.writerows(dataset.get("operation_examples", ()))

    metadata = {
        "scenario": config.scenario.value,
        "seed": config.seed,
        "input_low": config.input_low,
        "input_high": config.input_high,
        "noise_std": config.noise_std,
        "operation_noise": None if config.operation_noise is None else dataclasses.asdict(config.operation_noise),
        "readout_noise": None if config.readout_noise is None else dataclasses.asdict(config.readout_noise),
        "train_fraction": config.train_fraction,
        "protocols": [
            {
                "name": protocol.name,
                "steps": list(protocol.steps),
                "n_samples": protocol.n_samples,
                "role": protocol.role.value,
                "semantic_variant": protocol.semantic_variant,
                "input_low": config.input_low if protocol.input_low is None else protocol.input_low,
                "input_high": config.input_high if protocol.input_high is None else protocol.input_high,
                "observed_operations": (
                    None if protocol.observed_operations is None else list(protocol.observed_operations)
                ),
                "observed_step_indices": (
                    None if protocol.observed_step_indices is None else list(protocol.observed_step_indices)
                ),
                "missing_trace_rules": [
                    dataclasses.asdict(rule)
                    for rule in protocol.missing_trace_rules
                ],
            }
            for protocol in config.protocols
        ],
    }
    with (config.output_dir / "metadata.yaml").open("w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def generate_and_write(config: BenchmarkConfig) -> dict[str, list[dict[str, str]]]:
    """Generate a dataset and write it to the configured output directory."""

    dataset = generate_dataset(config)
    write_dataset(config, dataset)
    return dataset


def _evaluate_protocol(x: np.ndarray, steps: Sequence[str], semantic_variant: str = "default") -> float:
    state = x.astype(float).copy()
    for step in steps:
        state = _apply_step(state, step, semantic_variant=semantic_variant)
    return _readout(state, semantic_variant=semantic_variant, steps=steps)


def _evaluate_protocol_with_trace(
    x: np.ndarray,
    steps: Sequence[str],
    protocol_name: str,
    protocol_role: ProtocolRole,
    sample_id: int,
    semantic_variant: str = "default",
    rng: np.random.Generator | None = None,
    operation_noise: NoiseConfig | None = None,
    readout_noise: NoiseConfig | None = None,
    legacy_noise_std: float = 0.01,
) -> tuple[float, list[dict[str, str]]]:
    state = x.astype(float).copy()
    trace = []
    for step_index, step in enumerate(steps):
        input_state = state.copy()
        next_state = _apply_step(state, step, semantic_variant=semantic_variant)
        next_state = _add_noise(next_state, operation_noise, rng)
        is_final_step = step_index == len(steps) - 1
        if is_final_step:
            y = _readout(next_state, semantic_variant=semantic_variant, steps=steps)
            y = float(_add_readout_noise(y, next_state, readout_noise, rng, legacy_noise_std=legacy_noise_std))
            output = np.array([y], dtype=float)
        else:
            output = next_state
            state = next_state
        trace.append(
            {
                "protocol": protocol_name,
                "protocol_role": protocol_role.value,
                "semantic_variant": semantic_variant,
                "sample_id": str(sample_id),
                "step_index": str(step_index),
                "operation": step,
                "input_1": _format_float(float(input_state[0])),
                "input_2": _format_float(float(input_state[1])),
                "output_1": _format_float(float(output[0])),
                "output_2": "" if output.shape[0] == 1 else _format_float(float(output[1])),
                "output_dim": str(output.shape[0]),
            }
        )
        if is_final_step:
            return float(y), trace
    raise ValueError("Protocol steps must not be empty.")


def _add_readout_noise(
    value: float,
    state: np.ndarray,
    noise: NoiseConfig | None,
    rng: np.random.Generator | None,
    legacy_noise_std: float,
) -> float:
    if noise is None:
        if rng is None or legacy_noise_std <= 0.0:
            return float(value)
        return float(value + rng.normal(0.0, legacy_noise_std))
    noisy = _add_noise(np.array([value], dtype=float), noise, rng, reference=state)
    return float(noisy[0])


def _add_noise(
    value: np.ndarray,
    noise: NoiseConfig | None,
    rng: np.random.Generator | None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    if noise is None or rng is None:
        return np.asarray(value, dtype=float)
    scale = _noise_scale(reference if reference is not None else value, noise)
    if scale <= 0.0:
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float) + _sample_noise(rng, size=value.shape, scale=scale, noise=noise)


def _noise_scale(reference: np.ndarray, noise: NoiseConfig) -> float:
    if noise.kind == "heteroscedastic":
        ref = np.asarray(reference, dtype=float).ravel()
        center = np.asarray(noise.center[: ref.shape[0]], dtype=float)
        if center.shape[0] < ref.shape[0]:
            center = np.pad(center, (0, ref.shape[0] - center.shape[0]), constant_values=0.0)
        distance2 = float(np.sum((ref - center) ** 2))
        bandwidth = max(noise.bandwidth, 1e-12)
        weight = float(np.exp(-0.5 * distance2 / (bandwidth**2)))
        return noise.base_std + noise.peak_std * weight
    return noise.std


def _sample_noise(
    rng: np.random.Generator,
    size: tuple[int, ...],
    scale: float,
    noise: NoiseConfig,
) -> np.ndarray:
    if noise.kind == "heavy_tailed":
        if noise.distribution == "laplace":
            return rng.laplace(0.0, scale / np.sqrt(2.0), size=size)
        df = max(noise.df, 1.01)
        variance = df / (df - 2.0) if df > 2.0 else 1.0
        standardized_scale = scale / np.sqrt(variance)
        return rng.standard_t(df, size=size) * standardized_scale
    return rng.normal(0.0, scale, size=size)


def _readout(
    state: np.ndarray,
    semantic_variant: str = "default",
    steps: Sequence[str] = (),
) -> float:
    return semantic_readout(state, semantic_variant=semantic_variant, steps=steps)


def _apply_step(state: np.ndarray, step: str, semantic_variant: str = "default") -> np.ndarray:
    return apply_semantic_step(state, step, semantic_variant=semantic_variant)


def _filter_observed_trace(
    trace: Sequence[dict[str, str]],
    protocol: ProtocolConfig,
) -> list[dict[str, str]]:
    examples = list(trace)
    if protocol.observed_operations is not None:
        observed = set(protocol.observed_operations)
        examples = [example for example in examples if example["operation"] in observed]
    if protocol.observed_step_indices is not None:
        observed_indices = {str(step_index) for step_index in protocol.observed_step_indices}
        examples = [example for example in examples if example["step_index"] in observed_indices]
    if protocol.missing_trace_rules:
        examples = [
            example
            for example in examples
            if not any(_trace_missing_rule_matches(example, rule) for rule in protocol.missing_trace_rules)
        ]
    return examples


def _trace_missing_rule_matches(example: Mapping[str, str], rule: TraceMissingRule) -> bool:
    if rule.operations is not None and example["operation"] not in set(rule.operations):
        return False
    if rule.step_indices is not None and int(example["step_index"]) not in set(rule.step_indices):
        return False
    input_1 = float(example["input_1"])
    input_2 = float(example["input_2"])
    if rule.input_1_low is not None and input_1 < rule.input_1_low:
        return False
    if rule.input_1_high is not None and input_1 > rule.input_1_high:
        return False
    if rule.input_2_low is not None and input_2 < rule.input_2_low:
        return False
    if rule.input_2_high is not None and input_2 > rule.input_2_high:
        return False
    return True


def _format_float(value: float) -> str:
    return f"{value:.10f}"
