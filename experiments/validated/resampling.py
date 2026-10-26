"""Full MBAR refits in independently resampled chain blocks."""
import math
import numpy as np
from .mbar import fit_mbar

def observables(fit,thresholds=(20.,30.,40.,60.)):
    w=fit.weights.normalized_weights;c=fit.weights.costs;n=fit.weights.steps
    atoms=fit.weights.atom_mask
    masks={"P(C<=10)":c<=10,"atom_6":atoms&(n==6),
           "continuous_bin_10_10.5":~atoms&(c>=10)&(c<10.5)}
    masks.update({f"P(C>={z:g})":c>=z for z in thresholds})
    result={"mean_C":float(w@c),"mean_N":float(w@n)}
    result.update({key:float(w@mask) for key,mask in masks.items()})
    return result

def block_indices(rng,n,block):
    if type(block) is not int or not 1<=block<=n:raise ValueError("block size must be in [1,n]")
    starts=rng.integers(0,n,size=(n+block-1)//block)
    return ((starts[:,None]+np.arange(block))%n).ravel()[:n]

def full_refit_bootstrap(windows,*,blocks=512,replicates=49,seed=2026101709,
                         thresholds=(20.,30.,40.,60.),fit=None):
    if type(replicates) is not int or not 20<=replicates<=1000:
        raise ValueError("20..1000 bootstrap replicates required")
    windows=tuple(windows);fit=fit or fit_mbar(windows)
    if type(blocks) is int:
        blocks=[1 if w.independent else blocks for w in windows]
    if len(blocks)!=len(windows):raise ValueError("one block size per independent chain")
    if any(type(b) is not int or not 1<=b<=len(w.costs)//4 for b,w in zip(blocks,windows)):
        raise ValueError("block size must permit at least four blocks per chain")
    rng=np.random.Generator(np.random.PCG64(seed))
    estimates=observables(fit,thresholds);rows=[];normalizers=[]
    for _ in range(replicates):
        resampled=[w.resampled(block_indices(rng,len(w.costs),block))
                   for w,block in zip(windows,blocks)]
        refit=fit_mbar(resampled,fit.weights.target,initial_f=fit.f)
        rows.append(observables(refit,thresholds));normalizers.append(refit.f)
    result={}
    c=fit.weights.costs
    for key,value in estimates.items():
        v=np.array([r[key] for r in rows])
        sparse=False;events=None;transitions=None
        masks=None
        if key.startswith("P(C>="):
            z=float(key[5:-1]);masks=[w.costs>=z for w in windows]
        elif key=="P(C<=10)":masks=[w.costs<=10 for w in windows]
        elif key=="atom_6":
            masks=[(w.steps==6)&((w.long_steps==0) if w.model.b>0 else True) for w in windows]
        elif key=="continuous_bin_10_10.5":
            masks=[(w.costs>=10)&(w.costs<10.5)&(w.long_steps>0)&(w.model.b>0) for w in windows]
        if masks is not None:
            events=sum(int(mask.sum()) for mask in masks)
            transitions=sum(int(np.count_nonzero(mask[1:]!=mask[:-1])) for mask in masks)
            sparse=transitions<20 or events==0
        se=float(np.std(v,ddof=1))
        result[key]={"estimate":value,"standard_error":None if sparse else se,
                     "percentile_interval_95":None if sparse else np.quantile(v,[.025,.975]).tolist(),
                     "bootstrap_sd_diagnostic":se,"events":events,"event_transitions":transitions,
                     "status":"unavailable_sparse_events" if sparse else "provisional_block_bootstrap",
                     "warning":"requires equilibrium, finite MCMC asymptotic variance, adequate blocks and model-specific calibration"}
    return {"method":"circular moving blocks separately within every chain; independent observations use block=1",
            "normalizers_refitted":True,"replicates":replicates,"blocks_in_retained_observations":list(blocks),
            "seed":seed,"rng":"NumPy PCG64 (analysis resampling, separate from physical simulation)",
            "observables":result,"f_standard_error":np.std(normalizers,axis=0,ddof=1).tolist()}
