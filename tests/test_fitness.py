import numpy as np

from pso_tfidf.config import FitnessConfig, PSOConfig
from pso_tfidf.fitness import build_fitness, clamp_params


def test_clamp_proportional():
    pso = PSOConfig()
    min_df, max_df = clamp_params(0.5, 0.2, pso)
    assert min_df < max_df
    assert pso.b_lo <= min_df <= pso.b_hi


def test_clamp_absolute_min_df():
    pso = PSOConfig()
    min_df, max_df = clamp_params(5, 0.8, pso)
    assert min_df == 5
    assert max_df == 0.8


def test_fitness_runs_on_tiny_corpus():
    corpus = ["cat dog"] * 40 + ["bird fish"] * 40
    labels = np.array([0] * 40 + [1] * 40)
    evaluate = build_fitness(
        corpus,
        labels,
        PSOConfig(population=4, max_iter=2),
        FitnessConfig(cv_folds=3, min_features=2),
    )
    cost, nf = evaluate(1, 0.99)  # absolute min_df=1 doc
    assert np.isfinite(cost)
    assert nf > 0
