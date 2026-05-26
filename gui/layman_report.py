"""
Plain-language reports and charts for non-technical users.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pso_tfidf.types import BenchmarkResult

METHOD_LABELS = {
    "fixed_default": "Standard TF-IDF (fixed settings)",
    "grid_search": "Grid search (tries a preset grid)",
    "random_search": "Random search (random trials)",
    "pso_tuned": "PSO tuned (swarm-optimized)",
}

GLOSSARY = """
GLOSSARY — terms used in this tool
----------------------------------

TF-IDF
  A way to turn text into numbers for machine learning. It weights words that
  are important in a document but not common in every document.

min_df / max_df
  Filters for how rare or common a word must be across the dataset to be kept.
  PSO searches for a good pair of these cutoffs.

PSO (Particle Swarm Optimization)
  A search strategy inspired by flocks of birds. Many candidate settings are
  tried and improved over iterations toward the best score.

Macro F1 (reported as "Classification score")
  A single number for how well topics/classes are separated (0% = poor, 100% = perfect).
  "Macro" means every class counts equally, even rare ones.

Cross-validation (CV)
  The dataset is split into several folds; each method is trained and tested on
  different folds. The reported score is the average — a fairer estimate than
  one lucky split.

Vocabulary size (features)
  How many distinct words remain after filtering. Too few can lose information;
  too many can add noise and slow training.

p-value (statistical test)
  If PSO is compared to the best baseline: values below 0.05 often mean the
  gap is unlikely to be pure luck. Above 0.05 means the difference could be chance.

Cohen's d (effect size)
  How large the practical gap is, not just whether it exists. Rough guide:
  |d| < 0.2 small, 0.2–0.5 medium, > 0.5 large.

Evaluation budget (n_evaluations)
  How many times each method scored a candidate setting. More comparisons =
  fairer when judging PSO vs random/grid search.
""".strip()


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def format_at_a_glance(result: BenchmarkResult) -> str:
    df = result.summary.copy()
    if df.empty:
        return "No results to display."

    winner = result.winner or "unknown"
    winner_row = df[df["method"] == winner].iloc[0]
    score_pct = float(winner_row["cv_f1_macro_mean"]) * 100
    score_std = float(winner_row["cv_f1_macro_std"]) * 100

    pso_row = df[df["method"] == "pso_tuned"]
    lines = [
        "AT A GLANCE",
        "===========",
        "",
        f"Best approach: {_label(winner)}",
        f"   Classification score: {score_pct:.1f}% (± {score_std:.1f}%)",
        "",
        "In plain terms:",
        "  The tool tested several ways to filter words before classification.",
        "  The winner above achieved the highest average score across cross-validation",
        "  folds on a sample of the 20 Newsgroups dataset (news posts by topic).",
        "",
    ]

    if not pso_row.empty:
        pso = pso_row.iloc[0]
        pso_pct = float(pso["cv_f1_macro_mean"]) * 100
        sorted_df = df.sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
        rank = int(sorted_df[sorted_df["method"] == "pso_tuned"].index[0]) + 1
        lines.extend(
            [
                f"PSO-tuned score: {pso_pct:.1f}% (rank #{rank} of {len(df)})",
                f"  • Words kept (vocabulary): about {int(pso['n_features']):,}",
                f"  • Search time: {float(pso['time_seconds']):.0f} seconds",
                "",
            ]
        )

    stats = result.stats
    if stats:
        base = _label(str(stats.get("baseline_method", "baseline")))
        lift = float(stats.get("f1_lift_mean", 0)) * 100
        p = float(stats.get("paired_ttest_pvalue", 1))
        d = float(stats.get("cohens_d_pso_minus_baseline", 0))
        lines.extend(
            [
                "PSO vs strongest baseline (" + base + "):",
                f"  • Average score difference: {lift:+.2f} percentage points",
                f"  • Statistical confidence (p-value): {p:.3f}",
                "    " + _interpret_pvalue(p),
                f"  • Practical effect size (Cohen's d): {d:.2f}",
                "    " + _interpret_cohens_d(d),
                "",
            ]
        )

    lines.extend(
        [
            "IMPORTANT DISCLAIMERS",
            "---------------------",
            "• Scores are for this dataset sample and settings — not universal truth.",
            "• Higher score ≠ perfect classification; news topics overlap in wording.",
            "• Longer PSO runs may find better settings; Quick demo is indicative only.",
            "• Results are saved on disk; closing the app does not delete them.",
            "",
            f"Full files: {result.results_dir}",
        ]
    )
    return "\n".join(lines)


def _interpret_pvalue(p: float) -> str:
    if p < 0.05:
        return "(Often taken as: PSO and baseline differ beyond random chance.)"
    return "(Often taken as: difference may be due to chance — not conclusive.)"


def _interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "(Small practical difference.)"
    if ad < 0.5:
        return "(Moderate practical difference.)"
    return "(Large practical difference.)"

def format_compare_methods(result: BenchmarkResult) -> str:
    df = result.summary.copy()
    if df.empty:
        return "No comparison table available."

    lines = [
        "METHOD COMPARISON (higher score is better)",
        "==========================================",
        "",
        f"{'Method':<42} {'Score':>7} {'±':>5} {'Words':>8} {'Time':>7}",
        "-" * 73,
    ]

    for _, row in df.iterrows():
        name = _label(str(row["method"]))[:40]
        pct = float(row["cv_f1_macro_mean"]) * 100
        std = float(row["cv_f1_macro_std"]) * 100
        nf = int(row["n_features"]) if np.isfinite(row["n_features"]) else 0
        t = float(row["time_seconds"])
        marker = " *" if str(row["method"]) == result.winner else ""
        lines.append(f"{name:<42} {pct:6.1f}% {std:5.1f} {nf:8,} {t:6.0f}s{marker}")

    lines.extend([
        "",
        "Column guide:",
        "  Score  — average classification quality across CV folds (like % correct balance).",
        "  ±      — how much the score varied across folds (lower = more stable).",
        "  Words  — vocabulary size after TF-IDF filtering.",
        "  Time   — wall-clock time for that method's search.",
        "  * — best score in this run.",
    ])
    return "\n".join(lines)


def format_what_it_means(result: BenchmarkResult) -> str:
    parts = [GLOSSARY, "", "YOUR RUN — HOW TO READ IT", "-" * 28, ""]

    if result.stats:
        p = float(result.stats.get("paired_ttest_pvalue", 1))
        lift = float(result.stats.get("f1_lift_mean", 0)) * 100
        base = _label(str(result.stats.get("baseline_method", "baseline")))
        parts.extend(
            [
                f"We compared PSO-tuned TF-IDF to {base}.",
                "",
                f"Average score lift: {lift:+.2f} percentage points.",
                _interpret_pvalue(p),
                "",
                "Do not over-read a small lift:",
                "  A +1 point gain can matter for tuning pipelines, but it does not mean",
                "  the model is production-ready or 'understands' language.",
                "",
            ]
        )
    else:
        parts.append("No head-to-head statistics were computed for this run.")

    df = result.summary
    if not df.empty and "pso_tuned" in df["method"].values:
        fixed = df[df["method"] == "fixed_default"]
        pso = df[df["method"] == "pso_tuned"]
        if not fixed.empty and not pso.empty:
            f_pct = float(fixed.iloc[0]["cv_f1_macro_mean"]) * 100
            p_pct = float(pso.iloc[0]["cv_f1_macro_mean"]) * 100
            parts.extend(
                [
                    "Standard vs PSO (if both ran):",
                    f"  Standard fixed settings: {f_pct:.1f}%",
                    f"  PSO tuned:               {p_pct:.1f}%",
                    "",
                ]
            )

    parts.extend(
        [
            "CHARTS (see Charts tab)",
            "  • Score comparison — bar heights = average quality; error bars = variability.",
            "  • Improvement chart — PSO vs best baseline side by side (if available).",
            "  • Optimization path — how PSO score changed each iteration.",
        ]
    )
    return "\n".join(parts)


def format_log_friendly(technical_lines: list[str]) -> str:
    """Rewrite common log lines for lay readers."""
    friendly = [
        "ACTIVITY LOG",
        "",
    ]
    mapping = [
        ("Downloading / loading 20 Newsgroups", "Loading the news dataset…"),
        ("Corpus ready:", "Text ready for analysis:"),
        ("Running baselines", "Testing standard and search baselines…"),
        ("Starting particle swarm", "Starting swarm optimization (PSO)…"),
        ("PSO iteration", "Optimization step"),
        ("Stagnation", "Search plateaued — shaking up part of the swarm"),
        ("Computing statistics", "Crunching numbers and drawing charts…"),
        ("Complete. Results saved", "All done! Files saved to"),
        ("fixed_default:", "Standard settings:"),
        ("grid_search:", "Grid search:"),
        ("random_search:", "Random search:"),
        ("PSO done:", "PSO finished:"),
    ]
    for line in technical_lines:
        out = line
        for old, new in mapping:
            if old in line:
                out = line.replace(old, new)
                break
        friendly.append(out)
    return "\n".join(friendly)


CHART_CAPTIONS = {
    "comparison_layman": (
        "Which method scored highest?",
        "Taller bars = better average classification. Error bars show consistency across folds."
    ),
    "improvement": (
        "PSO vs best baseline",
        "Side-by-side view of the two most comparable scores from this run."
    ),
    "convergence": (
        "How PSO improved over time",
        "Each step is one swarm iteration. Upward trend means the search found better word filters."
    ),
}


def build_layman_figures(result: BenchmarkResult, figures_dir: Path) -> dict[str, Path]:
    """Create additional charts with readable labels; return paths keyed for GUI."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    df = result.summary.copy()
    if df.empty:
        return out

    df["label"] = df["method"].map(_label)
    df["pct"] = df["cv_f1_macro_mean"] * 100
    df["pct_std"] = df["cv_f1_macro_std"] * 100

    # --- Horizontal scorecard ---
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(df))))
    colors = ["#4C9F70" if m == result.winner else "#5B7DB1" if m == "pso_tuned" else "#7A7E85" for m in df["method"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["pct"], xerr=df["pct_std"], color=colors, capsize=4, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"], fontsize=10)
    ax.set_xlabel("Classification score (%) — higher is better")
    ax.set_title("Method comparison (20 Newsgroups sample)", fontsize=13, pad=12)
    ax.set_xlim(0, min(100, df["pct"].max() + df["pct_std"].max() + 12)) 
    
    ax.axvline(df["pct"].max(), color="#E8B84A", linestyle="--", alpha=0.6, label="Best score")
    
    for i, (_, row) in enumerate(df.iterrows()):
        # 2. Changed row["pct"] + 0.5 to look at row["pct"] + row["pct_std"] + 1.0
        # This pushes the text completely past the black error bars.
        # Added ha="left" to keep the alignment consistent.
        offset = row["pct_std"] + 1.0
        ax.text(row["pct"] + offset, i, f"{row['pct']:.1f}%", va="center", ha="left", fontsize=9)
        
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "comparison_layman.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["comparison_layman"] = path

    # --- PSO vs best baseline ---
    stats = result.stats
    if stats and "pso_tuned" in df["method"].values:
        base_name = str(stats.get("baseline_method", ""))
        base_row = df[df["method"] == base_name]
        pso_row = df[df["method"] == "pso_tuned"]
        if not base_row.empty and not pso_row.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            names = [_label(base_name), _label("pso_tuned")]
            vals = [float(base_row.iloc[0]["cv_f1_macro_mean"]) * 100, float(pso_row.iloc[0]["cv_f1_macro_mean"]) * 100]
            bars = ax.bar(names, vals, color=["#7A7E85", "#5B7DB1"], width=0.5)
            ax.set_ylabel("Classification score (%)")
            ax.set_title("PSO vs best baseline in this run")
            ax.set_ylim(0, min(100, max(vals) * 1.15 + 5))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11)
            lift = float(stats.get("f1_lift_mean", 0)) * 100
            ax.annotate(
                f"Difference: {lift:+.2f} pts",
                xy=(0.5, 0.95),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                color="#333",
            )
            fig.tight_layout()
            path = figures_dir / "improvement.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            out["improvement"] = path

    if result.pso_history:
        scores = [-c for c in result.pso_history]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(scores) + 1), scores, color="#2E86AB", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Classification quality (higher is better)")
        ax.set_title("PSO search progress")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = figures_dir / "convergence_layman.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out["convergence_layman"] = path

    return out
