# PSO-TF-IDF: Swarm-Tuned Document Frequency Bounds

**Tune `min_df` and `max_df` for TF-IDF with Particle Swarm Optimization**, then measure whether the representation improves **20 Newsgroups** classification against fixed defaults, grid search, and random search.

Static TF-IDF thresholds are a weak point in many NLP pipelines: they are often guessed, corpus-blind, and misaligned with the downstream task. This project treats `(min_df, max_df)` as a **2D continuous search problem** and optimizes them with PSO using **stratified CV macro-F1**—the same family of metric used in the final benchmark.

## Quick start

Requires **Python 3.11+** (3.12 recommended; see `.venv` setup below).

```powershell
cd "d:\Project\code\GitHub Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Full benchmark (~tens of minutes depending on hardware)
python experiments/run_benchmark.py --config configs/default.yaml

# Fast smoke run (~few minutes)
python experiments/run_benchmark.py --config configs/fast.yaml
```

Outputs land in [`results/`](results/):

| File | Description |
|------|-------------|
| `benchmark_summary.csv` | CV F1, hyperparameters, runtime, evaluation budget per method |
| `benchmark_stats.json` | Paired t-test vs best baseline, Cohen's d |
| `figures/pso_convergence.png` | PSO cost curve |
| `figures/cv_f1_comparison.png` | Bar chart across methods |

## Methods compared

| Method | What it does |
|--------|----------------|
| `fixed_default` | `min_df=5`, `max_df=0.8`, `max_features=1500` (classic sklearn-style baseline) |
| `grid_search` | Exhaustive grid over proportional `min_df` / `max_df` |
| `random_search` | Same search space, random trials (budget ~ PSO eval count) |
| `pso_tuned` | PSO over continuous `(min_df, max_df)` with stagnation recovery |

## Project layout

```
├── configs/           # default and fast YAML profiles
├── experiments/       # CLI entrypoints
├── notebooks/         # Original exploratory notebook (legacy)
├── src/pso_tfidf/     # Library: data, fitness, PSO, baselines, benchmark
├── tests/
└── results/           # Generated artifacts (gitignored figures; CSV committed optionally)
```

## Core design choices

1. **Fitness aligned with evaluation** — PSO minimizes negative CV macro-F1 (logistic regression, saga), not a mixed clustering + single holdout split score.
2. **Fair baselines** — Grid and random search use the **same fitness function** as PSO.
3. **Cached fitness evaluations** — Repeated `(min_df, max_df)` pairs during PSO are not re-fit.
4. **Portable paths** — No hard-coded `D:\Project\...`; everything config-driven.

## Desktop app (plug-and-play GUI)

A desktop interface runs the full pipeline in the background and shows summary tables, statistics, live log, and charts.

```powershell
cd "d:\Project\code\GitHub Project"
pip install -r requirements-gui.txt
pip install -e .
python gui/app.py
```

Or double-click **`Launch GUI.bat`** (after installing dependencies).

| Profile | Use case |
|---------|----------|
| Quick demo | ~1–3 minutes, 600 documents |
| Full benchmark | ~15–45 minutes, 1500 documents |

### Build a standalone `.exe` (Windows)

```powershell
.\scripts\build_exe.ps1
```

Output: `dist\PSO-TF-IDF\PSO-TF-IDF.exe` — share the whole folder (not only the exe). First run needs internet once to download the 20 Newsgroups corpus.

**Note:** The bundled app is large (~hundreds of MB) because it includes scikit-learn and dependencies. For a smaller installer, consider a web UI (Streamlit) or a thin launcher that requires Python installed.

## Legacy notebook

[`PSO_TF_20newsgroup.ipynb`](PSO_TF_20newsgroup.ipynb) is the original monolithic experiment. Prefer `experiments/run_benchmark.py` for reproducible runs; the notebook can be updated to call `pso_tfidf` imports.

