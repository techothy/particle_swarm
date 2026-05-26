"""PSO-driven TF-IDF hyperparameter tuning for text classification."""

__version__ = "0.1.0"

from pso_tfidf.benchmark import run_benchmark
from pso_tfidf.config import load_config
from pso_tfidf.types import BenchmarkResult

__all__ = ["load_config", "run_benchmark", "BenchmarkResult", "__version__"]
