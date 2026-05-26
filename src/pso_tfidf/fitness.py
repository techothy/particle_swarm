from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pso_tfidf.config import FitnessConfig, PSOConfig


def clamp_params(
    min_df: float,
    max_df: float,
    pso: PSOConfig,
) -> tuple[float, float]:
    """Clamp proportional thresholds; leave absolute document counts (min_df >= 1) intact."""
    min_df = float(min_df)
    max_df = float(max_df)
    if min_df < 1.0:
        min_df = float(np.clip(min_df, pso.b_lo, pso.b_hi - pso.min_gap))
        max_df = float(np.clip(max_df, min_df + pso.min_gap, pso.b_hi))
    else:
        # Absolute min_df (document count): only cap max_df at 1.0, not at PSO upper bound
        max_df = float(np.clip(max_df, 0.05, 1.0))
    return min_df, max_df


def build_fitness(
    corpus: list[str],
    labels: np.ndarray,
    pso_cfg: PSOConfig,
    fit_cfg: FitnessConfig,
) -> Callable[[float, float], tuple[float, int]]:
    """
    Return a cost function(min_df, max_df) -> (cost, n_features).
    Lower cost is better. Cost is negative (fitness - penalty).
    """
    y = np.asarray(labels)
    cv = StratifiedKFold(
        n_splits=fit_cfg.cv_folds,
        shuffle=True,
        random_state=fit_cfg.random_state,
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver="saga",
        random_state=fit_cfg.random_state,
    )

    @lru_cache(maxsize=4096)
    def _cached_eval(min_df_key: float, max_df_key: float) -> tuple[float, int]:
        min_df, max_df = clamp_params(min_df_key, max_df_key, pso_cfg)
        if min_df >= max_df:
            return 1e6, 0

        try:
            X = TfidfVectorizer(min_df=min_df, max_df=max_df).fit_transform(corpus)
        except ValueError:
            return 1e6, 0

        n_features = X.shape[1]
        if n_features < fit_cfg.min_features or n_features > fit_cfg.max_features:
            return 1e6, n_features

        if fit_cfg.mode == "cv_f1":
            scores = cross_val_score(
                clf,
                X,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
            )
            fitness = float(scores.mean())
        else:
            raise ValueError(f"Unknown fitness mode: {fit_cfg.mode}")

        penalty = fit_cfg.vocab_penalty * np.log1p(n_features) / np.log1p(len(corpus))
        cost = -(fitness - penalty)
        return cost, n_features

    def evaluate(min_df: float, max_df: float) -> tuple[float, int]:
        # Round for cache stability without changing search much
        key = (round(float(min_df), 5), round(float(max_df), 5))
        return _cached_eval(key[0], key[1])

    evaluate.clear_cache = _cached_eval.cache_clear  # type: ignore[attr-defined]
    return evaluate
