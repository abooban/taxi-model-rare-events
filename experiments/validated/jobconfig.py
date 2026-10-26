"""Strict execution configuration layered on S2 ModelConfig / Bias / Ensemble."""
from copy import deepcopy
from pathlib import Path
import json
import math
from .config import ModelConfig, Bias, Ensemble, _object, _finite
from .bounds import cap_bound
from .streams import integer

METHODS=("direct","suffix","positional2","uniform2","conditional","conditional_sequential")
MAX_CONFIG_BYTES=16384

def default_config(method="direct"):
    return {
        "schema_version":1,"method":method,"model":ModelConfig().to_dict(),
        "bias":Bias(0).to_dict(),
        "rng":{"family":"pcg32-xsh-rr-64-32","seed":20261017,"stream":"auto"},
        "iterations":10000,"warmup":0,"replicate":0,"initial_state":"baseline",
        "ensemble":{"kind":"unrestricted","max_steps":None},
        "proposal":{"uniform_floor":.1,"position_scale":12.},
        "observations":{"stride":1,"chunk_records":65536,"thresholds":[20.,30.,40.,60.]},
        "limits":{"wall_seconds":30.,"shutdown_grace_seconds":2.,"max_storage_bytes":67108864},
        "diagnostics":{"status_seconds":1.,"cap_error_budget":1e-10,
                       "minimum_effective_samples":400.,"minimum_tail_transitions":20},
        "output_dir":"experiments/pilot/"+method+"-{run_id}",
        "scientific_status":"provisional_engineering_model"
    }

class JobConfig:
    def __init__(self,value):
        d=deepcopy(value)
        _object(d,default_config().keys())
        integer("schema_version",d["schema_version"],1,1)
        if d["method"] not in METHODS:raise ValueError("unknown sampling method")
        self.model=ModelConfig.from_dict(d["model"])
        self.bias=Bias.from_dict(d["bias"]);self.bias.validate_model(self.model)
        self.ensemble=Ensemble.from_dict(d["ensemble"])
        rng=_object(d["rng"],("family","seed","stream"))
        if rng["family"]!="pcg32-xsh-rr-64-32":raise ValueError("unsupported specified PRNG")
        integer("seed",rng["seed"],0,2**64-1)
        if rng["stream"]!="auto":integer("stream",rng["stream"],0,2**63-1)
        integer("iterations",d["iterations"],1,10**12)
        integer("warmup",d["warmup"],0,10**12)
        integer("replicate",d["replicate"],0,2**31-1)
        if d["initial_state"] not in ("baseline","low","high","long_N"):raise ValueError("invalid initial state")
        if d["scientific_status"]!="provisional_engineering_model":
            raise ValueError("these configurations carry no final scientific approval")
        prop=_object(d["proposal"],("uniform_floor","position_scale"))
        floor=_finite("uniform_floor",prop["uniform_floor"],positive=True)
        scale=_finite("position_scale",prop["position_scale"],positive=True)
        if not 1e-6<=floor<=1 or not 1e-3<=scale<=1e6:raise ValueError("positional parameters outside tested numeric domain")
        obs=_object(d["observations"],("stride","chunk_records","thresholds"))
        integer("stride",obs["stride"],1,10**9);integer("chunk_records",obs["chunk_records"],1,10**6)
        thresholds=obs["thresholds"]
        if not isinstance(thresholds,list) or not 1<=len(thresholds)<=64:raise ValueError("1..64 explicit thresholds required")
        obs["thresholds"]=[_finite("threshold",x,nonnegative=True) for x in thresholds]
        if any(b<=a for a,b in zip(obs["thresholds"],obs["thresholds"][1:])):raise ValueError("thresholds must strictly increase")
        limits=_object(d["limits"],("wall_seconds","shutdown_grace_seconds","max_storage_bytes"))
        for key in ("wall_seconds","shutdown_grace_seconds"):
            limits[key]=_finite(key,limits[key],positive=True)
        if not .01<=limits["wall_seconds"]<=86400 or not .05<=limits["shutdown_grace_seconds"]<=30:
            raise ValueError("runtime/grace outside finite-job limits")
        integer("max_storage_bytes",limits["max_storage_bytes"],131072,2**40)
        diag=_object(d["diagnostics"],("status_seconds","cap_error_budget","minimum_effective_samples","minimum_tail_transitions"))
        for key in ("status_seconds","cap_error_budget","minimum_effective_samples"):
            diag[key]=_finite(key,diag[key],positive=True)
        if diag["status_seconds"]<.01 or not 0<diag["cap_error_budget"]<1:
            raise ValueError("invalid status interval or cap error budget")
        integer("minimum_tail_transitions",diag["minimum_tail_transitions"],1,10**9)
        conditional=d["method"].startswith("conditional")
        if (d["method"]=="direct" or conditional) and (self.bias.beta!=0 or d["warmup"]!=0 or d["initial_state"]!="baseline"):
            raise ValueError("independent methods require beta=0, zero warmup and baseline initialization")
        fixed=d["method"] in ("positional2","uniform2")
        if fixed != (self.ensemble.kind=="finite_M"):raise ValueError("method / ensemble mismatch")
        self.cap=None
        if fixed:
            integer("max_steps",self.ensemble.max_steps,2,10**6)
            self.cap=cap_bound(self.ensemble.max_steps,self.bias.beta,self.model)
            if self.cap["total_variation_upper_bound"]>diag["cap_error_budget"]:
                raise ValueError("finite-M cap bound exceeds the declared error budget")
        mu=self.model.lambda_*self.model.L
        mean=(1+mu)*(1+self.model.b/self.model.lambda_*math.exp(-self.model.lambda_*self.model.eta_c))
        if not math.isfinite(mu) or not 1e-9<=mu<=1e6 or not math.isfinite(mean):
            raise ValueError("physical parameters outside the tested execution numeric domain")
        out=d["output_dir"]
        if not isinstance(out,str) or not out or len(out)>1000 or "\x00" in out:
            raise ValueError("explicit output directory required")
        if "{" in out.replace("{run_id}","") or "}" in out.replace("{run_id}",""):
            raise ValueError("only {run_id} output placeholder is supported")
        self.data=d

    @classmethod
    def read(cls,path):
        if Path(path).stat().st_size>MAX_CONFIG_BYTES:
            raise ValueError("execution JSON exceeds the bounded 16 KiB configuration limit")
        def duplicate_check(items):
            result={}
            for key,value in items:
                if key in result:raise ValueError(f"duplicate JSON field: {key}")
                result[key]=value
            return result
        return cls(json.loads(Path(path).read_text(),object_pairs_hook=duplicate_check,
                              parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {x}"))))
