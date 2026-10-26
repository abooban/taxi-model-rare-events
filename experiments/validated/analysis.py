"""Audited run analysis and plots. Refuses incomplete or incompatible inputs."""
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import numpy as np
from .data import read_observations
from .config import ModelConfig, Ensemble, HistogramSpec, ReconstructionConfig
from .jobconfig import JobConfig
from .mbar import SamplingWindow, fit_mbar
from .reconstruct import summarize
from .conditional import iid_log_summary
from .statistics import (inefficiency,indicator_diagnostics,chain_comparison,burnin_diagnostics,
                         ratio_influence,wilson_interval)
from .resampling import full_refit_bootstrap,observables
from .build import ROOT
from .config import _object

def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [json_safe(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    return value

def run_diagnostics(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    cfg=JobConfig(manifest["effective_config"]);d=cfg.data
    a=read_observations(directory)
    if len(a)<16:raise ValueError("at least 16 retained observations for diagnostics")
    if d["method"].startswith("conditional"):
        return {"run_id":manifest["run_id"],"method":d["method"],"retained":len(a),
                "tails":iid_log_summary(a["log_tails"]),"mean_C":inefficiency(a["mean_C"]),
                "uncertainty":"iid prefix contributions; overshoot analytically integrated"}
    values={"cost":a["C"],"N":a["N"]}
    result={key:inefficiency(v) for key,v in values.items()}
    for z in d["observations"]["thresholds"]:
        result[f"C>={z:g}"]=indicator_diagnostics(a["C"]>=z,d["diagnostics"]["minimum_tail_transitions"])
    result["atom"]=indicator_diagnostics((a["long_steps"]==0) if cfg.model.b>0 else np.ones(len(a),bool))
    return {"run_id":manifest["run_id"],"method":d["method"],"beta":cfg.bias.beta,
            "retained":len(a),"observables":result,
            "burnin":{key:burnin_diagnostics(v,d["warmup"],d["observations"]["stride"]) for key,v in values.items()},
            "uncertainty":"observable-specific; a passing diagnostic does not certify equilibrium"}

def conditional_report(directories):
    arrays=[];batches=[];settings=[];ids=[];streams=[]
    for directory in directories:
        root=Path(directory);m=json.loads((root/"manifest.json").read_text())
        cfg=JobConfig(m["effective_config"])
        if not cfg.data["method"].startswith("conditional"):raise ValueError("conditional runs required")
        if m["test_injection"]:raise ValueError("fault-injected data cannot be used for inference")
        a=read_observations(root)
        arrays.append(a["log_tails"].copy());batches.append(iid_log_summary(a["log_tails"]))
        settings.append((cfg.model,cfg.data["observations"]["thresholds"]))
        ids.append(m["run_id"]);streams.append(m["stream_allocation"]["stream"])
    if len(set(ids))!=len(ids) or len(set(streams))!=len(streams):
        raise ValueError("iid replications must have distinct runs and registered streams")
    if not settings or any(s!=settings[0] for s in settings):raise ValueError("conditional physics/thresholds differ")
    pooled=iid_log_summary(np.concatenate(arrays))
    between=None
    if len(batches)>=2 and len({b["count"] for b in batches})==1:
        means=np.array([b["estimate"] for b in batches])
        between=(np.std(means,axis=0,ddof=1)/math.sqrt(len(means))).tolist()
    return {"model":settings[0][0].to_dict(),"thresholds":settings[0][1],
            "run_ids":ids,"registered_streams":streams,"replicate_count":len(batches),
            "pooled_iid":pooled,"independent_batches":batches,"between_batch_standard_error":between,
            "interpretation":"Monte Carlo estimates of a remaining prefix expectation; not exact finite-sample probabilities",
            "calibration_limit":"surrogate calibration and independent-reference agreement do not prove physical rare-prefix coverage"}

def reconstruct_report(directories,*,edges=None,blocks=(128,256),replicates=39,seed=2026101713,analysis_stride=1):
    if type(analysis_stride) is not int or analysis_stride<1:raise ValueError("positive analysis stride required")
    if not isinstance(blocks,(list,tuple)) or not blocks or any(type(b) is not int or b<1 for b in blocks):
        raise ValueError("explicit positive integer block sizes required")
    raw_bytes=sum(p.stat().st_size for d in directories for p in Path(d).glob("chunk-*.bin"))
    if raw_bytes/analysis_stride>256*1024**2:
        raise MemoryError("pooled analysis storage exceeds 256 MiB; use an explicit analysis stride")
    windows=[]
    for directory in directories:
        w=SamplingWindow.from_run(directory)
        windows.append(w if analysis_stride==1 else w.resampled(np.arange(0,len(w.costs),analysis_stride)))
    target=ReconstructionConfig(windows[0].model,windows[0].ensemble,
                               HistogramSpec(tuple(edges) if edges is not None else tuple(np.arange(5,40.5,.5))))
    fit=fit_mbar(windows,target)
    mass=summarize(fit.weights)
    estimates=observables(fit)
    comparisons={}
    for beta in fit.betas:
        group=[w for w in windows if w.bias.beta==beta]
        comparisons[str(beta)]={key:chain_comparison([fn(w) for w in group]) for key,fn in {
            "cost":lambda w:w.costs,"N":lambda w:w.steps,
            "C>=20":lambda w:w.costs>=20,"C>=30":lambda w:w.costs>=30}.items()}
    influences={};offsets=np.cumsum([0,*fit.chain_counts])
    for key,indicator in {"P(C>=20)":fit.weights.costs>=20,
                          "P(C>=30)":fit.weights.costs>=30}.items():
        _,series=ratio_influence(indicator,fit.weights.log_weights)
        influences[key]=[inefficiency(series[offsets[i]:offsets[i+1]]) for i in range(len(windows))]
    boot=[full_refit_bootstrap(windows,blocks=int(block),replicates=replicates,
                              seed=seed+i,fit=fit) for i,block in enumerate(blocks)]
    unavailable=[key for key in estimates if all(
        b["observables"][key]["status"]=="unavailable_sparse_events" for b in boot)]
    return {"fit":fit.diagnostics,"estimates":{k:None if k in unavailable else v for k,v in estimates.items()},
            "raw_empirical_point_estimates":estimates,"unavailable_probability_estimates":unavailable,
            "analysis_stride_of_recorded_rows":analysis_stride,
            "mass_partition":{"edges":list(mass.histogram_edges),
                              "continuous_bin_mass":mass.bin_mass.tolist(),
                              "atom_mass":mass.atom_mass,"underflow_mass":mass.underflow_mass,
                              "overflow_mass":mass.overflow_mass,"total_mass":mass.total_mass,
                              "normalization":"MBAR gauge over full sample; no plotting-range renormalization"},
            "independent_chain_comparisons":comparisons,
            "fixed_f_ratio_influence_diagnostics":influences,
            "influence_limit":"fixed-f series omit normalizer-fitting error; final resampling refits all normalizers",
            "block_sensitivity":boot,
            "readiness":"provisional uncertainty pending physical-model calibration, adequate slow-observable diagnostics and target-host checks"}

def make_plots(report,directories,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    if report["mode"]=="conditional":
        r=report["result"];p=np.array(r["pooled_iid"]["estimate"]);se=np.array(r["pooled_iid"]["standard_error"])
        axes[0].errorbar(r["thresholds"],p,yerr=1.96*se,fmt="o",capsize=3)
        axes[0].set_yscale("log");axes[0].set(xlabel="Cost threshold z",ylabel="P(C ≥ z)",title="Conditional iid estimates (pilot errors)")
        for i,b in enumerate(r["independent_batches"]):
            axes[1].plot(r["thresholds"],b["estimate"],"o-",alpha=.7,label=f"Batch {i+1}")
        axes[1].set_yscale("log");axes[1].set(xlabel="Cost threshold z",ylabel="P(C ≥ z)",title="Independent prefix batches")
        axes[1].legend(fontsize=8)
    elif report["mode"]=="mbar":
        r=report["result"];mass=r["mass_partition"];edges=np.array(mass["edges"])
        axes[0].stairs(np.array(mass["continuous_bin_mass"])/np.diff(edges),edges,label="Continuous density")
        if mass["atom_mass"]:
            x=np.array([int(k) for k in mass["atom_mass"]]);y=np.array(list(mass["atom_mass"].values()))
            axes[0].stem(x,y,linefmt="C1-",markerfmt="C1o",basefmt=" ",label="Discrete mass")
        axes[0].set(xlabel="Cost",ylabel="Density / discrete atom mass",title="Full-crossing MBAR (separate measures)")
        axes[0].legend(fontsize=8)
        im=axes[1].imshow(r["fit"]["overlap_matrix"],vmin=0,vmax=1,cmap="viridis")
        fig.colorbar(im,ax=axes[1]);axes[1].set(xlabel="Window j",ylabel="Window i",title="MBAR overlap")
    else:
        for directory in directories[:4]:
            a=read_observations(directory)
            values=a["mean_C"] if "mean_C" in a.dtype.names else a["C"]
            axes[0].plot(a["index"][::max(1,len(a)//2000)],values[::max(1,len(a)//2000)],alpha=.5)
            axes[1].hist(values,bins=50,density=False,histtype="step")
        axes[0].set(xlabel="Proposal / prefix index",ylabel="Cost",title="Recorded states (including holding times)")
        axes[1].set(xlabel="Cost",ylabel="Raw retained counts",title="Diagnostic histogram")
    fig.savefig(Path(output)/"diagnostics.png",dpi=160);plt.close(fig)
    return matplotlib.__version__

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--mode",choices=("diagnostics","conditional","mbar"))
    parser.add_argument("--runs",nargs="+");parser.add_argument("--output")
    parser.add_argument("--blocks",nargs="+",type=int)
    parser.add_argument("--resamples",type=int)
    parser.add_argument("--stride",type=int)
    args=parser.parse_args()
    cfg={}
    if args.config:
        cfg=json.loads(Path(args.config).read_text())
        _object(cfg,("schema_version","enabled","mode","run_directories","output_dir",
                     "blocks_in_retained_rows","resamples","analysis_stride","require_unbiased_reference"))
        if cfg["schema_version"]!=1 or cfg["enabled"] is not True:
            raise ValueError("analysis configuration is disabled or unsupported")
        if cfg["require_unbiased_reference"] is not True:raise ValueError("unbiased reference is mandatory")
    args.mode=args.mode or cfg.get("mode","diagnostics")
    args.runs=args.runs or cfg.get("run_directories")
    args.output=args.output or cfg.get("output_dir")
    args.blocks=args.blocks or cfg.get("blocks_in_retained_rows",[128,256])
    args.resamples=cfg.get("resamples",39) if args.resamples is None else args.resamples
    args.stride=cfg.get("analysis_stride",1) if args.stride is None else args.stride
    if args.mode not in ("diagnostics","conditional","mbar") or not args.runs or not args.output:
        raise ValueError("explicit mode, nonempty runs and new output directory required")
    directories=[Path(d).resolve() for d in args.runs]
    out=Path(args.output).resolve()
    if not any(out!=p and out.is_relative_to(p) for p in (ROOT/"results/pilot",ROOT/"results/production",ROOT/"logs")):
        raise ValueError("analysis output must be new and inside results/pilot, results/production or logs")
    out.mkdir(parents=True,exist_ok=False)
    if args.mode=="conditional":result=conditional_report(directories)
    elif args.mode=="mbar":result=reconstruct_report(directories,blocks=args.blocks,replicates=args.resamples,analysis_stride=args.stride)
    else:result=[run_diagnostics(d) for d in directories]
    report={"schema_version":1,"mode":args.mode,"input_runs":[str(d) for d in directories],"result":result,
            "software":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform()}}
    report["software"]["matplotlib"]=make_plots(report,directories,out)
    with (out/"analysis.json").open("x") as f:json.dump(json_safe(report),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"analysis":str(out/"analysis.json"),"plot":str(out/"diagnostics.png")}))

if __name__=="__main__":main()
