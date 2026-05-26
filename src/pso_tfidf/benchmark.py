from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from pso_tfidf.baselines import run_baselines
from pso_tfidf.config import ProjectConfig, load_config
from pso_tfidf.data import preprocess_20newsgroups
from pso_tfidf.evaluate import cross_validated_f1, mcnemar_table
from pso_tfidf.fitness import build_fitness
from pso_tfidf.pso import run_pso
from pso_tfidf.types import BenchmarkResult
from statsmodels.stats.contingency_tables import mcnemar


def _progress(on_progress: Callable[[str], None] | None, msg: str) -> None:
    if on_progress:
        on_progress(msg)
    else:
        print(msg)


def _cost_to_f1(cost: float, penalty_approx: float = 0.0) -> float:
    """Invert minimization cost back to approximate F1 (ignores penalty)."""
    return max(0.0, -cost - penalty_approx)


def cv_f1_per_fold(
    texts: list[str],
    labels: np.ndarray,
    min_df: float,
    max_df: float,
    cv_folds: int,
    random_state: int,
    max_features: int | None = None,
) -> np.ndarray:
    y = np.asarray(labels)
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    clf = LogisticRegression(max_iter=1000, solver="saga", random_state=random_state)
    scores = []
    kwargs: dict = {"min_df": min_df, "max_df": max_df}
    if max_features is not None:
        kwargs["max_features"] = max_features
    for train_idx, test_idx in kf.split(texts, y):
        tr = [texts[i] for i in train_idx]
        te = [texts[i] for i in test_idx]
        vec = TfidfVectorizer(**kwargs)
        Xtr = vec.fit_transform(tr)
        Xte = vec.transform(te)
        clf.fit(Xtr, y[train_idx])
        scores.append(f1_score(y[test_idx], clf.predict(Xte), average="macro"))
    return np.array(scores, dtype=float)


def run_benchmark(
    config: ProjectConfig | None = None,
    config_path: str | Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    if config is None:
        config = load_config(config_path)

    results_dir = Path(config.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    _progress(on_progress, "Downloading / loading 20 Newsgroups corpus…")
    bundle = preprocess_20newsgroups(config.data)
    texts, labels = bundle.texts, np.asarray(bundle.labels)
    _progress(on_progress, f"Corpus ready: {len(texts)} documents, {len(set(labels))} classes")

    evaluate = build_fitness(texts, labels, config.pso, config.fitness)

    rows: list[dict] = []

    # --- Baselines ---
    _progress(on_progress, "Running baselines (fixed, grid, random)…")
    for baseline in run_baselines(texts, labels, config):
        f1_mean, f1_std = cross_validated_f1(
            texts,
            labels,
            baseline.min_df,
            baseline.max_df,
            cv_folds=config.fitness.cv_folds,
            random_state=config.fitness.random_state,
            max_features=config.raw_baseline_max_features if baseline.name == "fixed_default" else None,
        )
        rows.append(
            {
                "method": baseline.name,
                "min_df": baseline.min_df,
                "max_df": baseline.max_df,
                "n_features": baseline.n_features,
                "fitness_cost": baseline.cost,
                "cv_f1_macro_mean": f1_mean,
                "cv_f1_macro_std": f1_std,
                "time_seconds": baseline.elapsed_seconds,
                "n_evaluations": baseline.n_evaluations,
            }
        )
        _progress(
            on_progress,
            f"  {baseline.name}: min_df={baseline.min_df:.4f} max_df={baseline.max_df:.4f} CV F1={f1_mean:.4f}",
        )

    # --- PSO ---
    _progress(on_progress, "Starting particle swarm optimization…")
    pso_result = run_pso(
        evaluate,
        config.pso,
        verbose=on_progress is None,
        on_progress=on_progress,
    )
    pso_f1_mean, pso_f1_std = cross_validated_f1(
        texts,
        labels,
        pso_result.best_min_df,
        pso_result.best_max_df,
        cv_folds=config.fitness.cv_folds,
        random_state=config.fitness.random_state,
    )
    rows.append(
        {
            "method": "pso_tuned",
            "min_df": pso_result.best_min_df,
            "max_df": pso_result.best_max_df,
            "n_features": pso_result.feature_counts[-1] if pso_result.feature_counts else np.nan,
            "fitness_cost": pso_result.best_cost,
            "cv_f1_macro_mean": pso_f1_mean,
            "cv_f1_macro_std": pso_f1_std,
            "time_seconds": pso_result.elapsed_seconds,
            "n_evaluations": pso_result.n_evaluations,
        }
    )
    _progress(
        on_progress,
        f"PSO done: min_df={pso_result.best_min_df:.4f} max_df={pso_result.best_max_df:.4f} "
        f"CV F1={pso_f1_mean:.4f} ({pso_result.n_evaluations} evals, {pso_result.elapsed_seconds:.1f}s)",
    )

    _progress(on_progress, "Computing statistics and figures…")
    df = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    df.to_csv(results_dir / "benchmark_summary.csv", index=False)

    # Paired test: best baseline vs PSO
    baseline_df = df[df["method"] != "pso_tuned"].sort_values("cv_f1_macro_mean", ascending=False)
    best_baseline = baseline_df.iloc[0] if len(baseline_df) else None
    stats_out: dict = {}
    if best_baseline is not None:
        folds = config.fitness.cv_folds
        f1_base = cv_f1_per_fold(
            texts,
            labels,
            float(best_baseline["min_df"]),
            float(best_baseline["max_df"]),
            folds,
            config.fitness.random_state,
            max_features=config.raw_baseline_max_features
            if best_baseline["method"] == "fixed_default"
            else None,
        )
        f1_pso = cv_f1_per_fold(
            texts,
            labels,
            pso_result.best_min_df,
            pso_result.best_max_df,
            folds,
            config.fitness.random_state,
        )
        ttest = stats.ttest_rel(f1_pso, f1_base)
        diff = f1_pso - f1_base
        cohens_d = float(diff.mean() / diff.std(ddof=1)) if diff.std() > 0 else 0.0
        stats_out = {
            "baseline_method": str(best_baseline["method"]),
            "paired_ttest_statistic": float(ttest.statistic),
            "paired_ttest_pvalue": float(ttest.pvalue),
            "cohens_d_pso_minus_baseline": cohens_d,
            "f1_lift_mean": float(diff.mean()),
        }
        with open(results_dir / "benchmark_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2)

    figures: dict[str, Path] = {}
    if pso_result.history:
        plt.figure(figsize=(8, 4))
        plt.plot([-c for c in pso_result.history], color="darkgreen", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Approx. fitness (neg. cost)")
        plt.title("PSO convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = figures_dir / "pso_convergence.png"
        plt.savefig(path, dpi=150)
        plt.close()
        figures["convergence"] = path

    plt.figure(figsize=(9, 4))
    plt.bar(df["method"], df["cv_f1_macro_mean"], yerr=df["cv_f1_macro_std"], capsize=4)
    plt.ylabel("CV macro F1")
    plt.title("TF-IDF tuning methods (20 Newsgroups)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = figures_dir / "cv_f1_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    figures["comparison"] = path

    _progress(on_progress, f"Complete. Results saved to {results_dir.resolve()}")
    return BenchmarkResult(
        summary=df,
        stats=stats_out,
        results_dir=results_dir,
        figures=figures,
        pso_history=list(pso_result.history),
    )
