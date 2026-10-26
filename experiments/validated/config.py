"""Checked full-crossing model and common-baseline reconstruction configuration.

This boundary does not launch or configure the archived C programs.
JSON numbers are checked without permissive string/boolean coercion.
"""
from dataclasses import asdict, dataclass, field
import math
from numbers import Real


def _finite(name, value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} is outside finite binary64") from None
    if (not math.isfinite(converted) or (converted == 0 and value != 0)
            or (positive and converted <= 0) or (nonnegative and value < 0)):
        raise ValueError(f"invalid {name}: {value!r}")
    return converted


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"expected exactly these fields: {sorted(keys)}")
    return value


@dataclass(frozen=True)
class ModelConfig:
    lambda_: float = 1.0
    L: float = 10.0
    eta_c: float = 2.0
    b: float = 2.0
    terminal_convention: str = "full_crossing"

    def __post_init__(self):
        for name in ("lambda_", "L", "eta_c", "b"):
            object.__setattr__(self, name, _finite(
                name, getattr(self, name), positive=name in ("lambda_", "L"),
                nonnegative=name in ("eta_c", "b")))
        if self.terminal_convention != "full_crossing":
            raise ValueError("only the full_crossing terminal convention is implemented")

    def to_dict(self):
        result = asdict(self)
        result["lambda"] = result.pop("lambda_")
        return result

    @classmethod
    def from_dict(cls, value):
        data = dict(_object(value, ("lambda", "L", "eta_c", "b", "terminal_convention")))
        data["lambda_"] = data.pop("lambda")
        return cls(**data)


@dataclass(frozen=True)
class Ensemble:
    """finite_M means the conditional baseline P0(. | N <= M)."""
    kind: str = "unrestricted"
    max_steps: int | None = None

    def __post_init__(self):
        if self.kind == "unrestricted":
            if self.max_steps is not None:
                raise ValueError("unrestricted ensembles cannot have a physical N cap")
        elif self.kind == "finite_M":
            if type(self.max_steps) is not int or self.max_steps < 1:
                raise ValueError("finite_M requires an explicit positive integer max_steps")
        else:
            raise ValueError("ensemble must be unrestricted or finite_M")

    @property
    def normalizer_convention(self):
        return "Z_beta=E0[exp(-beta*C)]" if self.kind == "unrestricted" else (
            "z_beta_M=E0[exp(-beta*C)|N<=M]")

    @classmethod
    def from_dict(cls, value):
        return cls(**_object(value, ("kind", "max_steps")))


@dataclass(frozen=True)
class Bias:
    beta: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "beta", _finite("beta", self.beta))
        if self.beta != 0 and not math.isfinite(1.0 / self.beta):
            raise ValueError("theta reciprocal is not representable")

    @property
    def theta(self):
        """None serializes theta=infinity at beta=0; theta=0 is never valid."""
        return None if self.beta == 0 else 1.0 / self.beta

    @classmethod
    def from_theta(cls, theta):
        if theta is None:
            return cls(0.0)
        theta = _finite("theta", theta)
        if theta == 0:
            raise ValueError("theta=0 is undefined; use beta=0 / theta=null for direct MC")
        beta = 1.0 / theta
        if not math.isfinite(beta) or beta == 0:
            raise ValueError("theta reciprocal is not representable")
        return cls(beta)

    def validate_model(self, model):
        # Use a ratio, avoiding overflow in beta*b. At equality the overshoot
        # integral diverges. With b=0 every finite beta is admissible.
        if model.b > 0 and self.beta < 0 and -self.beta >= model.lambda_ / model.b:
            raise ValueError("full-crossing bias requires beta > -lambda/b")

    def to_dict(self):
        if self.theta is not None and not math.isfinite(self.theta):
            raise ValueError("theta reciprocal is not representable")
        return {"beta": self.beta, "theta": self.theta}

    @classmethod
    def from_dict(cls, value):
        data = _object(value, ("beta", "theta"))
        result = cls(data["beta"])
        theta = data["theta"]
        if result.beta == 0:
            if theta is not None:
                raise ValueError("beta=0 must use theta=null (infinite theta)")
        else:
            other = cls.from_theta(theta)
            if theta is None or not math.isclose(
                    result.beta, other.beta, rel_tol=1e-14, abs_tol=0):
                raise ValueError("beta and theta must satisfy beta=1/theta")
        return result


@dataclass(frozen=True)
class Provenance:
    source: str
    run_id: str

    def __post_init__(self):
        _text("source", self.source)
        _text("run_id", self.run_id)

    @classmethod
    def from_dict(cls, value):
        return cls(**_object(value, ("source", "run_id")))


@dataclass(frozen=True)
class HistogramSpec:
    """Finite edges; [left,right), with the final right edge included."""
    edges: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.edges, (tuple, list)):
            raise ValueError("histogram edges must be an explicit sequence")
        edges = tuple(_finite("edge", x, nonnegative=True) for x in self.edges)
        if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("histogram requires at least two strictly increasing finite edges")
        object.__setattr__(self, "edges", edges)

    def to_dict(self):
        return {"edges": list(self.edges),
                "closure": "left_closed_last_right_closed",
                "atoms": "separate_structural_integer_masses",
                "outside_range": "retain_underflow_and_overflow"}

    @classmethod
    def from_dict(cls, value):
        data = _object(value, ("edges", "closure", "atoms", "outside_range"))
        result = cls(data["edges"])
        if result.to_dict() != data:
            raise ValueError("unsupported histogram/atom/overflow convention")
        return result


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    ensemble: Ensemble
    bias: Bias
    provenance: Provenance
    log_normalizer: float
    normalizer_source: str

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig) or not isinstance(self.ensemble, Ensemble):
            raise ValueError("run requires checked model and ensemble")
        if not isinstance(self.bias, Bias) or not isinstance(self.provenance, Provenance):
            raise ValueError("run requires checked bias and provenance")
        self.bias.validate_model(self.model)
        z = _finite("log_normalizer", self.log_normalizer)
        _text("normalizer_source", self.normalizer_source)
        if self.bias.beta == 0 and z != 0:
            raise ValueError("unbiased baseline has log_normalizer=0, including finite_M")
        if self.bias.beta > 0 and z > 0 or self.bias.beta < 0 and z < 0:
            raise ValueError("normalizer sign is incompatible with positive cost")
        object.__setattr__(self, "log_normalizer", z)

    def to_dict(self):
        return {"model": self.model.to_dict(), "ensemble": asdict(self.ensemble),
                "normalizer_convention": self.ensemble.normalizer_convention,
                "bias": self.bias.to_dict(), "provenance": asdict(self.provenance),
                "log_normalizer": self.log_normalizer,
                "normalizer_source": self.normalizer_source}

    @classmethod
    def from_dict(cls, value):
        data = _object(value, ("model", "ensemble", "normalizer_convention", "bias",
                               "provenance", "log_normalizer", "normalizer_source"))
        result = cls(ModelConfig.from_dict(data["model"]),
                     Ensemble.from_dict(data["ensemble"]), Bias.from_dict(data["bias"]),
                     Provenance.from_dict(data["provenance"]), data["log_normalizer"],
                     data["normalizer_source"])
        if data["normalizer_convention"] != result.ensemble.normalizer_convention:
            raise ValueError("normalizer must use the declared baseline ensemble")
        return result


@dataclass(frozen=True)
class ReconstructionConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    ensemble: Ensemble = field(default_factory=Ensemble)
    histogram: HistogramSpec = field(default_factory=lambda: HistogramSpec((5, 10, 20, 40)))

    def __post_init__(self):
        if not isinstance(self.model, ModelConfig) or not isinstance(self.ensemble, Ensemble):
            raise ValueError("target requires checked model and ensemble")
        if not isinstance(self.histogram, HistogramSpec):
            raise ValueError("target requires explicit checked bins")

    def to_dict(self):
        return {"model": self.model.to_dict(), "ensemble": asdict(self.ensemble),
                "histogram": self.histogram.to_dict()}

    @classmethod
    def from_dict(cls, value):
        data = _object(value, ("model", "ensemble", "histogram"))
        return cls(ModelConfig.from_dict(data["model"]), Ensemble.from_dict(data["ensemble"]),
                   HistogramSpec.from_dict(data["histogram"]))
