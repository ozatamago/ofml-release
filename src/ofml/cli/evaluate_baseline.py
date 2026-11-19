"""Command line entry point for baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import yaml

from ofml.benchmarks.synthetic_protocol_transfer.metrics import evaluate_predictions
from ofml.models import ProtocolMeanBaseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the protocol mean baseline.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing train.csv and test.csv.")
    parser.add_argument("--target-protocol", help="Optional target protocol for target_protocol_rmse.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = pathlib.Path(args.dataset_dir)
    train_rows = _read_rows(dataset_dir / "train.csv")
    test_rows = _read_rows(dataset_dir / "test.csv")

    model = ProtocolMeanBaseline().fit(train_rows)
    predictions = model.predict(test_rows)
    reference_predictions = np.full(len(test_rows), model.global_mean_, dtype=float)
    results = evaluate_predictions(
        test_rows,
        predictions,
        reference_predictions=reference_predictions,
        target_protocol=args.target_protocol,
    )
    print(yaml.safe_dump(results, sort_keys=True))


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
