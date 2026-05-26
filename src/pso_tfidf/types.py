from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BenchmarkResult:
    summary: pd.DataFrame
    stats: dict
    results_dir: Path
    figures: dict[str, Path] = field(default_factory=dict)
    pso_history: list[float] = field(default_factory=list)

    @property
    def winner(self) -> str | None:
        if self.summary.empty:
            return None
        return str(self.summary.iloc[0]["method"])
