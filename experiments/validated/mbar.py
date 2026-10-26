"""Common-support binless MBAR with an explicit unbiased reference gauge.

Different caps/unknown supports are rejected. Common finite-M analysis targets
P0(.|N<=M), never an unrestricted law. Fitting normalizers does not certify
equilibrium, finite asymptotic MCMC variance, or tail coverage.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
from .config import ModelConfig, Ensemble, Bias, Provenance, RunConfig, ReconstructionConfig
from .reconstruct import (SampleWindow, balance_weights, validate_observations,
                          EnsembleMismatch, NumericalRangeError)

class ConvergenceError(ValueError): pass
class OverlapError(ValueError): pass

@dataclass(frozen=True)
class SamplingWindow:
    model: ModelConfig
    ensemble: Ensemble
    bias: Bias
    provenance: Provenance
    costs: np.ndarray
    steps: np.ndarray
    long_steps: np.ndarray
    independent: bool = False
    stream_identity: tuple | None = None

    def __post_init__(self):
        if not isinstance(self.model,ModelConfig) or not isinstance(self.ensemble,Ensemble):
            raise ValueError("checked physical model and explicit ensemble required")
        if not isinstance(self.bias,Bias) or not isinstance(self.provenance,Provenance):
            raise ValueError("checked bias and provenance required")
        if type(self.independent) is not bool:raise ValueError("independence flag must be boolean")
        if self.stream_identity is not None:
            if (not isinstance(self.stream_identity,tuple) or len(self.stream_identity)!=2 or
                self.stream_identity[0]!="pcg32-xsh-rr-64-32" or
                type(self.stream_identity[1]) is not int or not 0<=self.stream_identity[1]<2**63):
                raise ValueError("invalid registered stream identity")
        self.bias.validate_model(self.model)
        values=validate_observations(self.costs,self.steps,self.long_steps,self.model,self.ensemble)
        for name,value in zip(("costs","steps","long_steps"),values):
            object.__setattr__(self,name,value)

    def resampled(self,indices):
        return SamplingWindow(self.model,self.ensemble,self.bias,self.provenance,
                              self.costs[indices],self.steps[indices],self.long_steps[indices],
                              self.independent,self.stream_identity)

    @classmethod
    def from_run(cls,directory):
        from .data import read_observations
        from .jobconfig import JobConfig
        root=Path(directory);manifest=json.loads((root/"manifest.json").read_text())
        config=JobConfig(manifest["effective_config"])
        if config.data["method"].startswith("conditional"):
            raise EnsembleMismatch("conditional-prefix records are not physical-trip MBAR observations")
        if manifest["test_injection"]:raise ValueError("fault-injected job cannot be a scientific window")
        a=read_observations(root)
        return cls(config.model,config.ensemble,config.bias,
                   Provenance("validated_engine:"+manifest["source"]["sha256"],manifest["run_id"]),
                   a["C"],a["N"].astype(np.int64),a["long_steps"].astype(np.int64),
                   config.data["method"]=="direct",
                   (config.data["rng"]["family"],config.data["rng"]["stream"]))

@dataclass(frozen=True)
class MbarFit:
    weights: object
    betas: tuple
    f: np.ndarray
    state_counts: tuple
    iterations: int
    residual: float
    overlap: np.ndarray
    spectral_gap: float
    connected: bool
    chain_counts: tuple

    @property
    def diagnostics(self):
        return {"gauge":"f(beta=0)=0; f=-log(Z) for the declared common baseline",
                "betas":list(self.betas),"f":self.f.tolist(),"log_Z":(-self.f).tolist(),
                "state_counts":list(self.state_counts),"chain_counts":list(self.chain_counts),
                "iterations":self.iterations,"max_equation_residual":self.residual,
                "overlap_matrix":self.overlap.tolist(),"overlap_connected":self.connected,
                "overlap_spectral_gap":self.spectral_gap,
                "total_base_mass":float(np.exp(self.weights.log_total_mass)),
                "equilibrium_certified":False,
                "standalone_plus2_full_law_uncertainty":"prohibited: infinite ideal second moment",
                "uncertainty":"requires chain diagnostics and full-refit resampling"}

def fit_mbar(windows,target=None,*,tolerance=1e-10,max_iterations=10000,
             minimum_overlap=.001,require_connected=True,initial_f=None,
             max_memory_bytes=512*1024**2):
    windows=tuple(windows)
    if not windows or any(not isinstance(w,SamplingWindow) for w in windows):
        raise ValueError("at least one documented SamplingWindow required")
    if target is None:target=ReconstructionConfig(windows[0].model,windows[0].ensemble)
    if not isinstance(target,ReconstructionConfig):raise ValueError("checked target required")
    if len({w.provenance.run_id for w in windows})!=len(windows):
        raise ValueError("duplicate run IDs would double-count observations")
    stream_ids=[w.stream_identity for w in windows if w.stream_identity is not None]
    if len(set(stream_ids))!=len(stream_ids):
        raise ValueError("reused registered stream selectors cannot be treated as independent chains")
    if not math.isfinite(tolerance) or tolerance<=0 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError("invalid convergence controls")
    if not 0<minimum_overlap<1:raise ValueError("invalid overlap threshold")
    for w in windows:
        if w.model!=target.model or w.ensemble!=target.ensemble:
            raise EnsembleMismatch("MBAR requires one documented common support; mixed/unknown caps rejected")
    betas=tuple(sorted({w.bias.beta for w in windows}))
    if 0. not in betas:
        raise EnsembleMismatch("an unbiased component with retained observations is mandatory")
    reference=betas.index(0.)
    counts=np.array([sum(len(w.costs) for w in windows if w.bias.beta==b) for b in betas],dtype=np.int64)
    n=int(counts.sum());k=len(betas)
    if n*(k*5+8)*8>max_memory_bytes:raise MemoryError("MBAR memory ceiling; choose a documented recording stride")
    costs=np.concatenate([w.costs for w in windows])
    try:
        with np.errstate(over="raise",invalid="raise"):
            u=np.array(betas)[:,None]*costs
    except FloatingPointError as exc:raise NumericalRangeError("beta*C outside binary64") from exc
    f=np.zeros(k) if initial_f is None else np.array(initial_f,float,copy=True)
    if f.shape!=(k,) or not np.all(np.isfinite(f)):raise ValueError("invalid initial normalizers")
    f-=f[reference]
    logn=np.log(counts)[:,None]
    def equations(current):
        den=np.logaddexp.reduce(logn+current[:,None]-u,axis=0)
        new=-np.logaddexp.reduce(-u-den,axis=1)
        new-=new[reference]
        return new,den
    for iteration in range(1,max_iterations+1):
        new,_=equations(f)
        error=float(np.max(abs(new-f)));f=new
        if error<=tolerance:break
    else:raise ConvergenceError(f"MBAR failed after {max_iterations} iterations; residual={error:g}")
    new,den=equations(f);residual=float(np.max(abs(new-f)))
    if residual>tolerance*1.05 or not np.all(np.isfinite(f)):
        raise ConvergenceError("MBAR final equation residual failed")
    # W_i integrates to one. O_ij = n_j sum_a W_ia W_ja is row-stochastic.
    W=np.exp(f[:,None]-u-den)
    gram=W@W.T
    overlap=gram*counts[None,:]
    symmetric=gram*np.sqrt(counts[:,None]*counts[None,:])
    eigenvalues=np.linalg.eigvalsh(symmetric)
    gap=1. if k==1 else float(1-eigenvalues[-2])
    reached={reference}
    while True:
        new_reached=reached|{j for i in reached for j in range(k)
                             if overlap[i,j]>=minimum_overlap and overlap[j,i]>=minimum_overlap}
        if new_reached==reached:break
        reached=new_reached
    connected=len(reached)==k
    if require_connected and (not connected or gap<1e-6):
        raise OverlapError("overlap is disconnected/near singular at the declared diagnostic threshold")
    checked=[]
    for w in windows:
        logz=-float(f[betas.index(w.bias.beta)])
        # Exact gauge/sign constraints; a tiny sign violation is not silently clipped.
        config=RunConfig(w.model,w.ensemble,w.bias,w.provenance,logz,"fitted common-support binless MBAR")
        checked.append(SampleWindow(config,w.costs,w.steps,w.long_steps))
    weights=balance_weights(checked,target)
    return MbarFit(weights,betas,f,tuple(int(x) for x in counts),iteration,residual,overlap,gap,connected,
                   tuple(len(w.costs) for w in windows))
