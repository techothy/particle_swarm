from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pso_tfidf.config import PSOConfig


@dataclass
class Particle:
    pos: list[float]
    pos_z: float
    velocity: np.ndarray
    best_pos: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.best_pos:
            self.best_pos = self.pos.copy()


@dataclass
class SwarmResult:
    best_min_df: float
    best_max_df: float
    best_cost: float
    history: list[float]
    feature_counts: list[int]
    elapsed_seconds: float
    n_evaluations: int


def _inertia(iteration: int, max_iterations: int, pso: PSOConfig) -> float:
    return pso.w_max - (pso.w_max - pso.w_min) * (iteration / max_iterations)


def run_pso(
    evaluate: Callable[[float, float], tuple[float, int]],
    pso: PSOConfig,
    verbose: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> SwarmResult:
    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        elif verbose:
            print(msg)
    rng = np.random.default_rng(pso.seed)
    np.random.seed(pso.seed)

    swarm_best: list[float] | None = None
    swarm_best_z = math.inf
    particles: list[Particle] = []
    history: list[float] = []
    feature_counts: list[int] = []
    n_evaluations = 0

    for _ in range(pso.population):
        x = rng.uniform(pso.b_lo, pso.b_hi - pso.min_gap)
        y = rng.uniform(x + pso.min_gap, pso.b_hi)
        z, nf = evaluate(x, y)
        n_evaluations += 1
        velocity = np.clip(rng.uniform(-pso.v_max, pso.v_max, size=2), pso.v_min, pso.v_max)
        p = Particle(pos=[x, y], pos_z=z, velocity=velocity)
        particles.append(p)
        if z < swarm_best_z:
            swarm_best_z = z
            swarm_best = p.pos.copy()

    assert swarm_best is not None
    stagnation = 0
    last_best = swarm_best_z
    start = time.perf_counter()

    for iteration in range(pso.max_iter):
        last_nf = 0
        for particle in particles:
            for dim in range(2):
                r1, r2 = rng.random(), rng.random()
                w = _inertia(iteration, pso.max_iter, pso)
                cognitive = pso.personal_c * r1 * (particle.best_pos[dim] - particle.pos[dim])
                social = pso.social_c * r2 * (swarm_best[dim] - particle.pos[dim])
                particle.velocity[dim] = np.clip(
                    w * particle.velocity[dim] + cognitive + social,
                    pso.v_min,
                    pso.v_max,
                )

            particle.pos[0] += particle.velocity[0]
            particle.pos[1] += particle.velocity[1]
            particle.pos[0] = float(np.clip(particle.pos[0], pso.b_lo, pso.b_hi - pso.min_gap))
            particle.pos[1] = float(
                np.clip(particle.pos[1], particle.pos[0] + pso.min_gap, pso.b_hi)
            )

            if rng.random() < pso.mutation_prob:
                particle.pos[0] += rng.normal(0, 0.03)
                particle.pos[1] += rng.normal(0, 0.03)
                particle.pos[0] = float(np.clip(particle.pos[0], pso.b_lo, pso.b_hi - pso.min_gap))
                particle.pos[1] = float(
                    np.clip(particle.pos[1], particle.pos[0] + pso.min_gap, pso.b_hi)
                )

            z, nf = evaluate(particle.pos[0], particle.pos[1])
            n_evaluations += 1
            last_nf = nf
            if not math.isfinite(z) or nf < 5:
                continue
            feature_counts.append(nf)

            best_z, _ = evaluate(particle.best_pos[0], particle.best_pos[1])
            if z < best_z:
                particle.best_pos = particle.pos.copy()
            if z < swarm_best_z:
                swarm_best = particle.pos.copy()
                swarm_best_z = z

        history.append(swarm_best_z)
        _emit(
            f"PSO iteration {iteration + 1}/{pso.max_iter}: cost={swarm_best_z:.4f} "
            f"min_df={swarm_best[0]:.4f} max_df={swarm_best[1]:.4f} features={last_nf}"
        )

        if abs(swarm_best_z - last_best) < 1e-6:
            stagnation += 1
        else:
            stagnation = 0
            last_best = swarm_best_z

        if stagnation >= pso.stagnation_patience:
            _emit(f"[INFO] Stagnation at iteration {iteration + 1}; reinitializing half the swarm.")
            for particle in particles[: pso.population // 2]:
                x = swarm_best[0] + rng.normal(0, 0.04)
                y = swarm_best[1] + rng.normal(0, 0.04)
                z, _ = evaluate(x, y)
                n_evaluations += 1
                particle.pos = [float(x), float(y)]
                particle.pos_z = z
                particle.best_pos = particle.pos.copy()
                particle.velocity = np.clip(
                    rng.uniform(-pso.v_max, pso.v_max, size=2), pso.v_min, pso.v_max
                )
            stagnation = 0

    elapsed = time.perf_counter() - start
    return SwarmResult(
        best_min_df=swarm_best[0],
        best_max_df=swarm_best[1],
        best_cost=swarm_best_z,
        history=history,
        feature_counts=feature_counts,
        elapsed_seconds=elapsed,
        n_evaluations=n_evaluations,
    )
