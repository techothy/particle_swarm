#!/usr/bin/env python
"""Run full benchmark: baselines + PSO + statistical comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pso_tfidf.benchmark import run_benchmark  # noqa: E402
from pso_tfidf.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="PSO-TF-IDF benchmark on 20 Newsgroups")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML config path",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_benchmark(config=config)
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
