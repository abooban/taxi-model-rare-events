"""Known-normalizer balance reconstruction for one explicit baseline ensemble.

D(c) = sum_k n_k exp(-beta_k*c) / Z_k
P_hat(B) = sum_a 1_B(c_a) / D(c_a).

For finite_M, Z_k here means z_beta,M relative to P0(.|N<=M).
Counts are actual retained rows, including rejection holding times, never ESS.
All absolute masses are retained in log form. Normalization is opt-in and uses
the complete observed support, including atoms and out-of-report-range rows.
No normalizer fitting, equilibrium certification or uncertainty is supplied.
"""
from dataclasses import dataclass
import math
import numpy as np

from .config import ReconstructionConfig, RunConfig, _finite


class EnsembleMismatch(ValueError):
    """Cost-only reconstruction cannot combine these sampling baselines."""


class NumericalRangeError(ValueError):
    """A requested linear result is outside binary64; inspect the log result."""


def logsumexp(values):
    a = np.asarray(values, dtype=float)
    if np.any(np.isnan(a)) or np.any(np.isposinf(a)):
        raise NumericalRangeError("log values must be finite or -infinity")
    if a.size == 0 or np.all(np.isneginf(a)):
        return -math.inf
    high = float(np.max(a))
    return high + math.log(float(np.exp(a - high).sum()))


def linear_mass(log_mass):
    a = np.asarray(log_mass, dtype=float)
    if np.any(np.isnan(a)) or np.any(np.isposinf(a)) or np.any(a > math.log(np.finfo(float).max)):
        raise NumericalRangeError("linear mass would overflow; use the reported log masses")
    with np.errstate(under="ignore"):
        result = np.exp(a)
    return float(result) if result.ndim == 0 else result


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _numbers(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a one-dimensional finite numeric sequence")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = _readonly(a, float)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable in binary64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be representable in binary64")
    return result


def _integers(values, name, *, positive=False):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector (no coercion or rounding)")
    if np.any(a > np.iinfo(np.int64).max) or np.any(a < (1 if positive else 0)):
        raise ValueError(f"invalid {name}")
    return _readonly(a, np.int64)


def _compatible(runs, target):
    if not isinstance(target, ReconstructionConfig):
        raise ValueError("expected checked ReconstructionConfig")
    for run in runs:
        if not isinstance(run, RunConfig):
            raise ValueError("every window needs a checked RunConfig")
        if run.model != target.model:
            raise EnsembleMismatch("model parameters/terminal convention differ from target")
        if run.ensemble != target.ensemble:
            raise EnsembleMismatch(
                "sampling baseline differs from target: unrestricted/finite_M and M "
                "must match; differing caps require C,N support treatment in T09")


def validate_observations(costs, steps, long_steps, model, ensemble):
    """Shared S2 physical-observation boundary for known-Z and fitted-Z inputs."""
    c = _numbers(costs, "costs")
    n = _integers(steps, "steps", positive=True)
    long = _integers(long_steps, "long_steps")
    if not len(c) or len(c) != len(n) or len(c) != len(long):
        raise ValueError("window must have equally sized, nonempty retained observations")
    if np.any(n > 2**53) or np.any(long > n) or np.any(c < n):
        raise ValueError("invalid cost, step count or above-threshold count")
    atoms = (long == 0) if model.b > 0 else np.ones(len(c), dtype=bool)
    if np.any(c[atoms] != n[atoms]):
        raise ValueError("structural integer atoms must have C=N exactly")
    if model.b > 0 and np.any(n[atoms].astype(np.longdouble) * model.eta_c <= model.L):
        raise ValueError("atom impossible: N*eta_c must be strictly greater than L")
    if ensemble.kind == "finite_M" and np.any(n > ensemble.max_steps):
        raise EnsembleMismatch("sample has N greater than the declared finite_M cap")
    return c, n, long


@dataclass(frozen=True)
class SampleWindow:
    config: RunConfig
    costs: np.ndarray
    steps: np.ndarray
    long_steps: np.ndarray

    def __post_init__(self):
        if not isinstance(self.config, RunConfig):
            raise ValueError("window requires a checked RunConfig with known log normalizer")
        c, n, long = validate_observations(self.costs, self.steps, self.long_steps,
                                          self.config.model, self.config.ensemble)
        for name, value in (("costs", c), ("steps", n), ("long_steps", long)):
            object.__setattr__(self, name, value)

    @property
    def atom_mask(self):
        return self.long_steps == 0 if self.config.model.b > 0 else np.ones(
            len(self.costs), dtype=bool)


@dataclass(frozen=True)
class StabilizedWeights:
    values: np.ndarray
    log_shift: float
    underflow_count: int

    @property
    def restored_log_total(self):
        if not np.any(self.values > 0):
            raise NumericalRangeError("chosen shift underflowed all weights; use original logs")
        return logsumexp(np.log(self.values[self.values > 0])) + self.log_shift


def stabilize(log_weights, log_shift=None):
    """w=exp(s)*scaled_w; normalized weights cancel the same shift.

    Underflow in a linear display is counted; balance reports always group the
    original logs, preserving small masses even if their display weights vanish.
    """
    logs = _numbers(log_weights, "log_weights")
    if not len(logs):
        raise ValueError("weights must be nonempty")
    shift = float(np.max(logs)) if log_shift is None else _finite("log_shift", log_shift)
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = logs - shift
    values = linear_mass(adjusted)
    return StabilizedWeights(_readonly(values, float), shift,
                             int(np.count_nonzero(values == 0)))


@dataclass(frozen=True)
class BalanceWeights:
    target: ReconstructionConfig
    costs: np.ndarray
    steps: np.ndarray
    atom_mask: np.ndarray
    log_weights: np.ndarray
    runs: tuple[RunConfig, ...]
    window_counts: tuple[int, ...]

    @property
    def log_total_mass(self):
        return logsumexp(self.log_weights)

    @property
    def normalized_weights(self):
        return linear_mass(self.log_weights - self.log_total_mass)

    def stabilized(self, log_shift=None):
        return stabilize(self.log_weights, log_shift)


def _log_denominator(costs, runs, counts):
    den = np.full(len(costs), -np.inf)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for run, count in zip(runs, counts):
                term = math.log(int(count)) - run.bias.beta * costs - run.log_normalizer
                if not np.all(np.isfinite(term)):
                    raise FloatingPointError
                den = np.logaddexp(den, term)
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalRangeError("beta*C/log denominator outside binary64") from exc
    if not np.all(np.isfinite(den)):
        raise NumericalRangeError("nonfinite log denominator")
    return den


def balance_weights(windows, target):
    windows = tuple(windows)
    if not windows or any(not isinstance(w, SampleWindow) for w in windows):
        raise ValueError("provide at least one nonempty SampleWindow")
    runs = tuple(w.config for w in windows)
    _compatible(runs, target)
    counts = tuple(len(w.costs) for w in windows)
    costs = _readonly(np.concatenate([w.costs for w in windows]), float)
    steps = _readonly(np.concatenate([w.steps for w in windows]), np.int64)
    atoms = _readonly(np.concatenate([w.atom_mask for w in windows]), bool)
    logs = _readonly(-_log_denominator(costs, runs, counts), float)
    return BalanceWeights(target, costs, steps, atoms, logs, runs, counts)


def raw_count_log_masses(raw_counts, sample_counts, runs, target, representative_costs):
    """Binned reference: log[sum H_kj / sum n_k exp(-beta_k*c_j)/Z_k].

    This is exact only when each bias is constant in the category. On continuous
    finite bins, representative_costs introduces a bin-center approximation.
    It returns category probability masses, not densities or a renormalized law.
    """
    runs = tuple(runs)
    if not runs:
        raise ValueError("provide at least one run")
    _compatible(runs, target)
    ns = _integers(sample_counts, "sample_counts", positive=True)
    c = _numbers(representative_costs, "representative_costs")
    h = np.asarray(raw_counts)
    if not len(c) or np.any(c < 0) or h.ndim != 2 or h.shape != (len(runs), len(c)):
        raise ValueError("raw count shape or category costs invalid")
    if h.dtype.kind not in "iu" or np.any(h < 0) or np.any(h > np.iinfo(np.int64).max):
        raise ValueError("raw_counts must be nonnegative integers, not preweighted counts")
    if len(ns) != len(runs):
        raise ValueError("sample_counts must give actual retained rows for every run")
    if any(sum(int(v) for v in row) > int(n) for row, n in zip(h, ns)):
        raise ValueError("raw histogram count exceeds actual retained rows")
    totals = np.array([sum(int(v) for v in h[:, j]) for j in range(len(c))], dtype=float)
    numerator = np.full(len(c), -np.inf)
    mask = totals > 0
    numerator[mask] = np.log(totals[mask])
    return numerator - _log_denominator(c, runs, ns)


@dataclass(frozen=True)
class MassReport:
    histogram_edges: tuple[float, ...]
    normalization: str
    bin_log_mass: np.ndarray
    atom_log_mass: dict[int, float]
    underflow_log_mass: float
    overflow_log_mass: float
    total_log_mass: float
    source_total_log_mass: float
    retained_count: int
    window_counts: tuple[int, ...]

    @property
    def bin_mass(self):
        return linear_mass(self.bin_log_mass)

    @property
    def continuous_log_density(self):
        return self.bin_log_mass - np.log(np.diff(self.histogram_edges))

    @property
    def continuous_density(self):
        return linear_mass(self.continuous_log_density)

    @property
    def atom_mass(self):
        return {n: linear_mass(v) for n, v in self.atom_log_mass.items()}

    @property
    def underflow_mass(self):
        return linear_mass(self.underflow_log_mass)

    @property
    def overflow_mass(self):
        return linear_mass(self.overflow_log_mass)

    @property
    def total_mass(self):
        return linear_mass(self.total_log_mass)

    @property
    def reported_mass(self):
        return linear_mass(logsumexp([*self.bin_log_mass, *self.atom_log_mass.values()]))

    @property
    def diagnostics(self):
        try:
            source_total = linear_mass(self.source_total_log_mass)
        except NumericalRangeError:
            source_total = None
        categories = np.array([*self.bin_log_mass, *self.atom_log_mass.values(),
                               self.underflow_log_mass, self.overflow_log_mass])
        small = np.isfinite(categories) & (categories < math.log(np.nextafter(0.0, 1.0)))
        return {"normalization": self.normalization,
                "absolute_log_total_mass": self.source_total_log_mass,
                "absolute_total_mass": source_total,
                "absolute_mass_error": None if source_total is None else source_total - 1,
                "linear_underflow_categories": int(np.count_nonzero(small)),
                "partition_log_mass": logsumexp(categories),
                "retained_count": self.retained_count,
                "window_counts": list(self.window_counts),
                "uncertainty": "not estimated; serial correlation and fitted-Z errors excluded"}


def summarize(weights, *, normalize=False):
    """Partition atoms, continuous bins, underflow and overflow without losing mass.

    normalize=True creates a finite-sample ratio estimator over ALL observed
    rows. It never conditions on the reporting interval. The default retains
    the absolute known-normalizer estimator and its normalization error.
    """
    if not isinstance(weights, BalanceWeights):
        raise ValueError("expected BalanceWeights")
    if type(normalize) is not bool:
        raise ValueError("normalize must be an explicit boolean")
    edges = weights.target.histogram.edges
    c = weights.costs
    inside = (c >= edges[0]) & (c <= edges[-1])
    source_total = weights.log_total_mass
    offset = source_total if normalize else 0.0

    def mass(mask):
        return logsumexp(weights.log_weights[mask]) - offset

    continuous = inside & ~weights.atom_mask
    ids = np.minimum(np.searchsorted(edges, c, side="right") - 1, len(edges) - 2)
    bins = _readonly([mass(continuous & (ids == j)) for j in range(len(edges) - 1)], float)
    atoms = {int(n): mass(inside & weights.atom_mask & (weights.steps == n))
             for n in np.unique(weights.steps[inside & weights.atom_mask])}
    return MassReport(tuple(edges), "self_normalized_full_sample" if normalize else
                      "absolute_known_normalizer", bins, atoms, mass(c < edges[0]),
                      mass(c > edges[-1]), source_total - offset, source_total,
                      len(c), weights.window_counts)
