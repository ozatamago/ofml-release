"""Command line entry point for benchmark generation."""

from __future__ import annotations

import argparse

from ofml.benchmarks.synthetic_protocol_transfer.generator import generate_and_write, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an OFML benchmark dataset.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    dataset = generate_and_write(config)
    print(
        f"Wrote {len(dataset['train'])} train rows and {len(dataset['test'])} test rows "
        f"to {config.output_dir}"
    )
