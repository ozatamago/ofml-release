"""Adapters from ofplang YAML protocols to OFML protocol configs."""

from __future__ import annotations

import pathlib
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

from ofml.benchmarks.synthetic_protocol_transfer.generator import ProtocolConfig


DEFAULT_EXCLUDE_TYPES = frozenset({"ServePlate96", "StoreLabware"})
DEFAULT_EXCLUDE_IDS = frozenset({"input", "output"})


def protocol_config_from_yaml(
    path: str | pathlib.Path,
    n_samples: int,
    name: str | None = None,
    step_map: Mapping[str, str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
) -> ProtocolConfig:
    """Convert an ofplang YAML protocol into a synthetic benchmark protocol."""

    protocol_path = pathlib.Path(path)
    with protocol_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("ofplang protocol YAML must be a mapping.")

    steps = steps_from_protocol_mapping(
        raw,
        step_map=step_map,
        exclude_types=exclude_types,
        exclude_ids=exclude_ids,
    )
    return ProtocolConfig(
        name=name or protocol_path.stem,
        steps=steps,
        n_samples=n_samples,
    )


def steps_from_protocol_mapping(
    raw: Mapping[str, Any],
    step_map: Mapping[str, str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Extract a stable topological operation sequence from an ofplang mapping."""

    operations = _read_operations(raw)
    operation_ids = tuple(operations.keys())
    excluded_types = set(DEFAULT_EXCLUDE_TYPES if exclude_types is None else exclude_types)
    excluded_ids = set(DEFAULT_EXCLUDE_IDS if exclude_ids is None else exclude_ids)
    ordered_ids = _topological_operation_ids(raw, operation_ids)

    steps = []
    for operation_id in ordered_ids:
        operation_type = operations[operation_id]
        if operation_id in excluded_ids or operation_type in excluded_types:
            continue
        steps.append(_map_step(operation_id, operation_type, step_map or {}))

    if not steps:
        raise ValueError("No benchmark steps were extracted from the ofplang protocol.")
    return tuple(steps)


def _read_operations(raw: Mapping[str, Any]) -> dict[str, str]:
    operations_raw = raw.get("operations")
    if not isinstance(operations_raw, Sequence) or isinstance(operations_raw, str):
        raise ValueError("ofplang protocol field 'operations' must be a sequence.")

    operations: dict[str, str] = {}
    for item in operations_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each operation entry must be a mapping.")
        operation_id = str(item["id"])
        operations[operation_id] = str(item["type"])
    return operations


def _topological_operation_ids(raw: Mapping[str, Any], operation_ids: Sequence[str]) -> tuple[str, ...]:
    operation_set = set(operation_ids)
    order_index = {operation_id: index for index, operation_id in enumerate(operation_ids)}
    incoming_count = {operation_id: 0 for operation_id in operation_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)

    for source, target in _operation_edges(raw):
        if source not in operation_set or target not in operation_set:
            continue
        outgoing[source].append(target)
        incoming_count[target] += 1

    ready = deque(
        sorted(
            (operation_id for operation_id, count in incoming_count.items() if count == 0),
            key=order_index.__getitem__,
        )
    )
    ordered = []

    while ready:
        operation_id = ready.popleft()
        ordered.append(operation_id)
        for target in sorted(outgoing[operation_id], key=order_index.__getitem__):
            incoming_count[target] -= 1
            if incoming_count[target] == 0:
                ready.append(target)

    if len(ordered) != len(operation_ids):
        raise ValueError("ofplang protocol operation graph contains a cycle.")
    return tuple(ordered)


def _operation_edges(raw: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    connections_raw = raw.get("connections", ())
    if not isinstance(connections_raw, Sequence) or isinstance(connections_raw, str):
        raise ValueError("ofplang protocol field 'connections' must be a sequence.")

    edges = []
    for item in connections_raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each connection entry must be a mapping.")
        source_address = item["input"]
        target_address = item["output"]
        if not _is_address(source_address) or not _is_address(target_address):
            raise ValueError("Each connection address must be a two-item sequence.")
        edges.append((str(source_address[0]), str(target_address[0])))
    return tuple(edges)


def _is_address(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2


def _map_step(operation_id: str, operation_type: str, step_map: Mapping[str, str]) -> str:
    return str(step_map.get(operation_id, step_map.get(operation_type, operation_type)))
