"""Observable-specific dependence diagnostics and explicit uncertainty limits."""
import math
from statistics import NormalDist
import numpy as np

def inefficiency(values):
    """Geyer initial-positive/monotone paired sequence, conservative g>=1.

    g=1+2 sum rho; ESS=n/g. Constant observables have unavailable ESS.
    """
    a=np.asarray(values,float)
    if a.ndim!=1 or len(a)<8 or not np.all(np.isfinite(a)):
        raise ValueError("at least eight finite observations required")
    n=len(a);x=a-a.mean();variance=float(x@x/n)
    if variance<=0:
        return {"g":None,"ess":None,"mean":float(a.mean()),"standard_error":None,
                "last_lag":0,"warning":"constant observable; information not established"}
    size=1<<(2*n-1).bit_length()
    f=np.fft.rfft(x,n=size)
    ac=np.fft.irfft(f*f.conjugate(),n=size)[:n//2]
    ac/=ac[0]
    limit=(len(ac)//2)*2
    pairs=ac[:limit:2]+ac[1:limit:2]
    stops=np.flatnonzero(pairs<=0)
    end=int(stops[0]) if len(stops) else len(pairs)
    g=max(1.,float(-1+2*np.minimum.accumulate(pairs[:end]).sum())) if end else 1.
    warning="lag exhaustion" if not len(stops) else ("short relative to correlation length" if n<100*g else None)
    return {"g":g,"ess":n/g,"mean":float(a.mean()),
            "standard_error":float(np.std(a,ddof=1)*math.sqrt(g/n)),
            "last_lag":max(0,2*end-1),"warning":warning}

def indicator_diagnostics(values,minimum_transitions=20):
    a=np.asarray(values)
    if a.ndim!=1 or not np.all((a==0)|(a==1)):raise ValueError("binary indicator required")
    result=inefficiency(a)
    transitions=int(np.count_nonzero(a[1:]!=a[:-1]))
    result.update(transitions=transitions,events=int(a.sum()),count=len(a))
    if transitions<minimum_transitions:
        result.update(ess=None,standard_error=None,warning="insufficient event transitions for relative uncertainty")
    return result

def _rhat(chains):
    m,n=chains.shape
    W=float(np.var(chains,axis=1,ddof=1).mean())
    B=float(n*np.var(chains.mean(axis=1),ddof=1))
    if W==0:return None if B==0 else math.inf
    return math.sqrt(((n-1)/n*W+B/n)/W)

def _rank_normalize(a):
    values,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts);ranks=ends-(counts-1)/2
    normal=NormalDist()
    z=np.array([normal.inv_cdf((r-3/8)/(a.size+1/4)) for r in ranks])
    return z[inverse].reshape(a.shape)

def chain_comparison(chains):
    chains=[np.asarray(c,float) for c in chains]
    if len(chains)<2 or min(map(len,chains))<16:
        return {"rank_split_rhat":None,"folded_rank_split_rhat":None,"warning":"two independent chains of length >=16 required"}
    half=min(map(len,chains))//2
    split=np.array([part for c in chains for part in (c[:half],c[-half:])])
    rank=_rhat(_rank_normalize(split))
    folded=_rhat(_rank_normalize(abs(split-np.median(split))))
    return {"rank_split_rhat":rank,"folded_rank_split_rhat":folded,
            "diagnostic_retained_per_split":half,
            "warning":"observable-specific diagnostic; does not prove equilibrium/tail coverage"}

def burnin_diagnostics(values,warmup,stride=1):
    a=np.asarray(values,float);half=len(a)//2
    if half<8:raise ValueError("at least sixteen post-warmup observations required")
    first=inefficiency(a[:half]);last=inefficiency(a[-half:]);whole=inefficiency(a)
    se1=first["standard_error"];se2=last["standard_error"]
    se=None if se1 is None or se2 is None else math.hypot(se1,se2)
    z=None if se is None or se==0 else (float(a[:half].mean()-a[-half:].mean()))/se
    return {"first_last_standardized_drift":z,
            "warmup_in_estimated_correlation_units":None if whole["g"] is None else warmup/(stride*whole["g"]),
            "warning":"retained-segment stationarity check; burn-in is not certified by this diagnostic"}

def ratio_influence(values,log_weights):
    """Fixed-normalizer ratio influence only; fitted-Z uncertainty is additional."""
    a=np.asarray(values,float);logs=np.asarray(log_weights,float)
    if a.shape!=logs.shape or a.ndim!=1 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(logs)):
        raise ValueError("finite aligned series required")
    w=np.exp(logs-np.max(logs));estimate=float(np.dot(w,a)/w.sum())
    influence=w*(a-estimate)/w.mean()
    return estimate,influence

def wilson_interval(events,count,z=1.959963984540054):
    if not 0<=events<=count or count<=0:raise ValueError("invalid binomial counts")
    p=events/count;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [0. if events==0 else max(0.,center-half),
            1. if events==count else min(1.,center+half)]
