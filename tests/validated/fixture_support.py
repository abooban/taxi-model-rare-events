"""Exact fixture builders, independent of the reconstruction algebra."""
from pathlib import Path
import json
import math
from fractions import Fraction
import numpy as np
from experiments.validated.config import (
    Bias, Ensemble, HistogramSpec, ModelConfig, Provenance, ReconstructionConfig, RunConfig)
from experiments.validated.reconstruct import SampleWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).parent / "fixtures/legacy_estimator_case.json").read_text())
COSTS = np.array(FIXTURE["costs"], dtype=float)
TRUTH = np.array([float(Fraction(p)) for p in FIXTURE["probability_fractions"]])
TARGET = ReconstructionConfig(histogram=HistogramSpec((5, 10, 20)))
RUNS = tuple(RunConfig(
    TARGET.model, TARGET.ensemble,
    Bias(w["beta_log2_over_2_multiplier"] * math.log(2) / 2),
    Provenance("exact_rational_discrete_fixture", f"window-{i}"),
    math.log(float(Fraction(w["normalizer_fraction"]))), "exact rational sum")
    for i, w in enumerate(FIXTURE["windows"]))
COUNTS = np.array([w["sample_count"] for w in FIXTURE["windows"]], dtype=int)
H = np.array([w["raw_counts"] for w in FIXTURE["windows"]], dtype=int)


def exact_windows():
    windows = []
    for run, h in zip(RUNS, H):
        c = np.repeat(COSTS, h)
        windows.append(SampleWindow(run, c, c.astype(int), np.zeros(len(c), dtype=int)))
    return windows


def run(beta=0.0, log_z=0.0, *, model=None, ensemble=None, run_id="test"):
    return RunConfig(model or ModelConfig(), ensemble or Ensemble(), Bias(beta),
                     Provenance("deterministic_test", run_id), log_z, "supplied exact test law")
