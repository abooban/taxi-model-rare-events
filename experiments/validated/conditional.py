"""Independent Python equations for the C conditional-prefix estimator."""
import math
import numpy as np
from .config import ModelConfig, Bias

def expected_cost(model=ModelConfig()):
    return (1 + model.lambda_*model.L) * (
        1 + model.b/model.lambda_*math.exp(-model.lambda_*model.eta_c))

def log_tail(pre, remainder, threshold, model=ModelConfig(), *, strict=False):
    pre, a, z = np.broadcast_arrays(np.asarray(pre, float), np.asarray(remainder, float),
                                     np.asarray(threshold, float))
    if not (np.all(np.isfinite(pre)) and np.all(pre >= 0) and
            np.all(np.isfinite(a)) and np.all((a > 0) & (a <= model.L)) and
            np.all(np.isfinite(z))):
        raise ValueError("invalid prefix, remainder or threshold")
    base = pre + 1 + model.b*np.maximum(a-model.eta_c, 0)
    if model.b == 0:
        return np.where(base > z if strict else base >= z, 0., -np.inf)
    residual = np.maximum((z-base)/model.b + np.maximum(model.eta_c-a, 0), 0)
    return np.where(z < base if strict else z <= base, 0., -model.lambda_*residual)

def tilted_prefix(pre, remainder, steps, beta, model=ModelConfig()):
    """Return log Z contribution, conditional tilted C mean, and N.

    Ratios of sample averages require joint covariance, not independent errors.
    """
    Bias(beta).validate_model(model)
    pre, a, n = np.broadcast_arrays(np.asarray(pre,float), np.asarray(remainder,float), steps)
    if model.b == 0:
        loga = np.full_like(pre, -beta)
        terminal_mean = np.ones_like(pre)
    else:
        rate = model.lambda_ + beta*model.b
        d = np.maximum(model.eta_c-a, 0)
        short = -np.expm1(-model.lambda_*d)
        long = np.exp(-model.lambda_*d) * model.lambda_/rate
        total = short + long
        loga = -beta - beta*model.b*np.maximum(a-model.eta_c,0) + np.log(total)
        terminal_mean = (short + long*(1+model.b/rate))/total + model.b*np.maximum(a-model.eta_c,0)
    return -beta*pre+loga, pre+terminal_mean, np.asarray(n,float)

def iid_log_summary(log_values):
    """Scaled iid covariance survives tiny probabilities without squaring them.

    Returns log estimates plus covariance and SE in linear binary64 when possible.
    """
    a = np.asarray(log_values,float)
    if a.ndim == 1: a = a[:,None]
    if a.ndim != 2 or len(a) < 2 or np.any(np.isnan(a)) or np.any(np.isposinf(a)):
        raise ValueError("at least two iid observations, finite logs or -infinity")
    maxima = np.max(a, axis=0)
    maxima = np.where(np.isfinite(maxima), maxima, 0)
    scaled = np.exp(a-maxima)
    means = scaled.mean(axis=0)
    covariance = np.atleast_2d(np.cov(scaled, rowvar=False, ddof=1))/len(a)
    scales = np.exp(maxima)
    with np.errstate(divide="ignore"):
        logs = np.log(means)+maxima
        log_se = np.log(np.sqrt(np.diag(covariance)))+maxima
    relative_se = np.divide(np.sqrt(np.diag(covariance)), means,
                            out=np.full_like(means,np.nan), where=means>0)
    return {"count": len(a), "log_estimate": logs.tolist(),
            "estimate": (means*scales).tolist(),
            "standard_error": (np.sqrt(np.diag(covariance))*scales).tolist(),
            "log_standard_error": log_se.tolist(),
            "linear_underflow": ((means>0)&(means*scales==0)).tolist(),
            "relative_standard_error": [float(x) if np.isfinite(x) else None for x in relative_se],
            "covariance_of_mean": (covariance*scales[:,None]*scales[None,:]).tolist(),
            "unavailable": (means == 0).tolist(),
            "interpretation": "iid finite-sample estimate; zero contributions provide no relative precision evidence"}
