"""
main.py — Entry point for the MedPACS-AI ETL pipeline.

Usage:
    python main.py
    python main.py --config path/to/custom_config.yaml
"""

import argparse

from etl.pipeline import MedPacsETLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedPACS-AI: Medical Imaging ETL Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    pipeline = MedPacsETLPipeline(config_path=args.config)
    pipeline.run()
