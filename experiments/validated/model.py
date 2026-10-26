"""Deterministic full-crossing evaluation only; no random generator or sampler."""
from dataclasses import dataclass
import math

from .config import ModelConfig, _finite


@dataclass(frozen=True)
class Evaluation:
    cost: float
    steps: int
    distance: float
    long_steps: int


def evaluate(increments, model=ModelConfig()):
    """Charge the whole crossing step; never silently accept an incomplete trip."""
    if not isinstance(model, ModelConfig):
        raise ValueError("expected checked ModelConfig")
    distance = cost = 0.0
    steps = long_steps = 0
    for value in increments:
        x = _finite("increment", value, positive=True)
        distance += x
        cost += 1.0 + model.b * max(x - model.eta_c, 0.0)
        steps += 1
        long_steps += x > model.eta_c
        if not math.isfinite(distance) or not math.isfinite(cost):
            raise ValueError("trajectory evaluation overflow")
        if distance >= model.L:
            return Evaluation(cost, steps, distance, long_steps)
    raise ValueError("incomplete trajectory: no full crossing of L")
