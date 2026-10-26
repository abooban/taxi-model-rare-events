"""Conservative analytical finite-M bounds, evaluated in logarithms."""
import math
from .config import ModelConfig, Bias
from .conditional import expected_cost

def log_poisson_upper(m, mu):
    if type(m) is not int or m < 0 or not math.isfinite(mu) or mu < 0:
        raise ValueError("invalid Poisson bound")
    if m == 0: return 0.
    if mu == 0: return -math.inf
    if m <= mu-1: return 0.
    return min(0., -mu+m*math.log(mu)-math.lgamma(m+1)-math.log1p(-mu/(m+1)))

def cap_bound(M, beta, model=ModelConfig()):
    if type(M) is not int or M < 1: raise ValueError("positive integer M required")
    Bias(beta).validate_model(model)
    mu = model.lambda_*model.L
    try:
        tilted_mu = mu*math.exp(-beta)
        if beta >= 0:
            log_num = -beta+mu*math.expm1(-beta)+log_poisson_upper(M,tilted_mu)
        else:
            t = -beta
            log_num = (t*model.b*model.L+t+mu*math.expm1(t)
                       -math.log1p(-t*model.b/model.lambda_)
                       +log_poisson_upper(M,tilted_mu))
        log_bound = min(0., log_num+beta*expected_cost(model))
    except (OverflowError, ValueError):
        log_bound = 0.
    # Separate unbounded-cost control: total variation alone cannot bound E C.
    moment_log=None;mean_error_log=None
    if log_bound<0:
        try:
            nu=mu*math.exp(-beta)
            if beta>=0:
                rate=model.lambda_
                log_A=-beta+mu*math.expm1(-beta)
            else:
                rate=model.lambda_+beta*model.b
                log_A=(-beta*model.b*model.L-beta+mu*math.expm1(-beta)
                       -math.log1p(beta*model.b/model.lambda_))
            B=1+model.b*(model.L+1/rate)
            q=math.log(B)+log_poisson_upper(M,nu)
            r=math.log(nu)+log_poisson_upper(M-1,nu)
            high=max(q,r)
            moment_log=log_A+high+math.log(math.exp(q-high)+math.exp(r-high))+beta*expected_cost(model)
            v_log=log_A+math.log(B+nu)+beta*expected_cost(model)
            other=log_bound+v_log-math.log1p(-math.exp(log_bound))
            high=max(moment_log,other)
            mean_error_log=high+math.log(math.exp(moment_log-high)+math.exp(other-high))
        except (OverflowError,ValueError,ZeroDivisionError):
            pass
    def linear_or_none(value):
        return math.exp(value) if value is not None and value<709 else None
    return {"M": M, "beta": beta, "log_total_variation_upper_bound": log_bound,
            "total_variation_upper_bound": math.exp(log_bound),
            "baseline_log_omitted_upper_bound": log_poisson_upper(M,mu),
            "log_omitted_cost_moment_upper_bound":moment_log,
            "log_mean_cost_difference_upper_bound":mean_error_log,
            "mean_cost_difference_upper_bound":linear_or_none(mean_error_log),
            "method": "Poisson geometric upper bound / Jensen lower bound; ordinary binary64 rounding"}
