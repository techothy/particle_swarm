from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from pso_tfidf.config import BaselineConfig, PSOConfig, ProjectConfig
from pso_tfidf.fitness import build_fitness, clamp_params


@dataclass
class BaselineResult:
    name: str
    min_df: float
    max_df: float
    cost: float
    n_features: int
    elapsed_seconds: float
    n_evaluations: int


def _search(
    name: str,
    candidates: list[tuple[float, float]],
    evaluate,
    pso: PSOConfig,
) -> BaselineResult:
    best = None
    start = time.perf_counter()
    n_eval = 0
    for min_df, max_df in candidates:
        min_df, max_df = clamp_params(min_df, max_df, pso)
        if min_df >= max_df:
            continue
        cost, nf = evaluate(min_df, max_df)
        n_eval += 1
        row = (cost, min_df, max_df, nf)
        if best is None or row[0] < best[0]:
            best = row
    if best is None:
        raise RuntimeError(f"No valid candidates for baseline {name}")
    elapsed = time.perf_counter() - start
    return BaselineResult(
        name=name,
        min_df=best[1],
        max_df=best[2],
        cost=best[0],
        n_features=best[3],
        elapsed_seconds=elapsed,
        n_evaluations=n_eval,
    )


def run_baselines(
    corpus: list[str],
    labels: np.ndarray,
    config: ProjectConfig,
) -> list[BaselineResult]:
    evaluate = build_fitness(corpus, labels, config.pso, config.fitness)
    pso = config.pso
    results: list[BaselineResult] = []

    # Fixed default-style baseline (sklearn-style absolute min_df count allowed)
    min_df = config.raw_baseline_min_df
    max_df = config.raw_baseline_max_df
    t0 = time.perf_counter()
    cost, nf = evaluate(float(min_df), float(max_df))
    results.append(
        BaselineResult(
            name="fixed_default",
            min_df=min_df,
            max_df=max_df,
            cost=cost,
            n_features=nf,
            elapsed_seconds=time.perf_counter() - t0,
            n_evaluations=1,
        )
    )

    grid_pairs = [
        (float(a), float(b))
        for a, b in itertools.product(
            config.baselines.grid_min_df,
            config.baselines.grid_max_df,
        )
        if a < b
    ]
    results.append(_search("grid_search", grid_pairs, evaluate, pso))

    rng = np.random.default_rng(config.pso.seed)
    random_pairs = []
    for _ in range(config.baselines.random_search_trials):
        x = rng.uniform(pso.b_lo, pso.b_hi - pso.min_gap)
        y = rng.uniform(x + pso.min_gap, pso.b_hi)
        random_pairs.append((x, y))
    results.append(_search("random_search", random_pairs, evaluate, pso))
    return results
