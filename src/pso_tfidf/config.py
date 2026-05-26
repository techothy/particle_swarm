from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PSOConfig:
    population: int = 25
    max_iter: int = 50
    v_max: float = 0.2
    v_min: float = -0.1
    personal_c: float = 1.5
    social_c: float = 2.5
    w_max: float = 0.9
    w_min: float = 0.4
    b_lo: float = 0.001
    b_hi: float = 0.4
    min_gap: float = 0.05
    mutation_prob: float = 0.2
    stagnation_patience: int = 8
    seed: int = 41


@dataclass
class FitnessConfig:
    cv_folds: int = 5
    min_features: int = 50
    max_features: int = 8000
    vocab_penalty: float = 0.02
    random_state: int = 42
    # "cv_f1" aligns optimization with downstream classification metrics
    mode: str = "cv_f1"


@dataclass
class DataConfig:
    max_docs: int = 1500
    test_size: float = 0.2
    subset: str = "all"


@dataclass
class BaselineConfig:
    grid_min_df: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02, 0.05])
    grid_max_df: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    random_search_trials: int = 125


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    results_dir: Path = field(default_factory=lambda: Path("results"))
    raw_baseline_min_df: float = 5
    raw_baseline_max_df: float = 0.8
    raw_baseline_max_features: int | None = 1500


def _merge_dict(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is None:
        return cfg

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if "data" in raw:
        cfg.data = DataConfig(**{**cfg.data.__dict__, **raw["data"]})
    if "pso" in raw:
        cfg.pso = PSOConfig(**{**cfg.pso.__dict__, **raw["pso"]})
    if "fitness" in raw:
        cfg.fitness = FitnessConfig(**{**cfg.fitness.__dict__, **raw["fitness"]})
    if "baselines" in raw:
        b = raw["baselines"]
        cfg.baselines = BaselineConfig(
            grid_min_df=b.get("grid_min_df", cfg.baselines.grid_min_df),
            grid_max_df=b.get("grid_max_df", cfg.baselines.grid_max_df),
            random_search_trials=b.get("random_search_trials", cfg.baselines.random_search_trials),
        )
    if "results_dir" in raw:
        cfg.results_dir = Path(raw["results_dir"])
    for key in ("raw_baseline_min_df", "raw_baseline_max_df", "raw_baseline_max_features"):
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg
