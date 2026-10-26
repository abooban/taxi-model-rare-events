"""Finite foreground job supervisor. Never launches a campaign or remote command."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import platform
import selectors
import signal
import subprocess
import sys
import time
import uuid
import numpy as np

from .build import ROOT, build, digest
from .data import verify_chunks, IntegrityError
from .jobconfig import JobConfig,MAX_CONFIG_BYTES
from .streams import register

METADATA_RESERVE=65536
LOG_LIMIT=16384

def metadata_reserve(config):
    return METADATA_RESERVE+max(0,len(config.data["observations"]["thresholds"])-4)*2048

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    with temporary.open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def git(*args):
    return subprocess.check_output(["git","--no-optional-locks","-C",str(ROOT),*args],text=True).strip()

def source_identity():
    paths=[]
    for folder in ("experiments/validated","tests/validated"):
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and
                     p.suffix in (".c",".h",".py",".json") and "__pycache__" not in p.parts)
    paths.extend(p for p in (ROOT/"scripts").glob("*validated*") if p.is_file())
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}
    content=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"revision":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),
            "dirty_status":git("status","--porcelain=v1","--untracked-files=all"),
            "sha256":content,"file_sha256":hashes}

def _safe_output(config,run_id):
    path=Path(config["output_dir"].replace("{run_id}",run_id))
    if not path.is_absolute():path=ROOT/path
    path=path.absolute()
    allowed=[ROOT/"experiments/pilot",ROOT/"experiments/production",ROOT/"logs"]
    resolved=path.resolve()
    if not any(resolved!=parent and resolved.is_relative_to(parent.resolve()) for parent in allowed):
        raise ValueError("run directory must be a new descendant of experiments/pilot, experiments/production or logs")
    if path.exists() or path.is_symlink():raise FileExistsError(f"refusing existing run directory: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.mkdir() # exclusive: a concurrent creation must fail
    return path

def _argv(config,folder,directory,fault_after_bytes):
    d=config.data;m=config.model;p=d["proposal"];obs=d["observations"];lim=d["limits"]
    values=[d["method"],m.lambda_,m.L,m.eta_c,m.b,config.bias.beta,d["iterations"],d["warmup"],
            obs["stride"],d["rng"]["seed"],d["rng"]["stream"],config.ensemble.max_steps or 0,
            p["uniform_floor"],p["position_scale"],d["initial_state"],lim["wall_seconds"],
            obs["chunk_records"],lim["max_storage_bytes"]-metadata_reserve(config),
            d["diagnostics"]["status_seconds"],str(directory),
            ",".join(format(x,".17g") for x in obs["thresholds"]),fault_after_bytes]
    return [str(folder/"engine"),*(str(x) for x in values)]

def _wait_test_child_ready(process,fd,received):
    """Bounded, test-only startup handshake; no production deadline is deferred."""
    deadline=time.monotonic()+5.
    with selectors.DefaultSelector() as ready:
        ready.register(fd,selectors.EVENT_READ)
        while process.poll() is None and not received:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            if ready.select(min(.02,remaining)):
                if os.read(fd,1)==b"R":return
                raise RuntimeError("test child readiness pipe closed or invalid marker")
    raise RuntimeError("test child did not become ready within startup deadline or was interrupted")

def run(value, *, original_bytes=None, registry=None, announce=False,
        _fault_after_bytes=0, _test_child=None, _test_child_ready=False, _sanitize=False):
    """All simulations run in C. Test-only injections are recorded in provenance."""
    config=JobConfig(value)
    if _test_child_ready and not _test_child:
        raise ValueError("readiness handshake is restricted to an injected test child")
    if original_bytes is not None and (not isinstance(original_bytes,bytes) or len(original_bytes)>MAX_CONFIG_BYTES):
        raise ValueError("exact input configuration must be at most 16 KiB")
    reserve=metadata_reserve(config)
    if config.data["limits"]["max_storage_bytes"]<=reserve+4096:
        raise ValueError("storage ceiling is insufficient for configuration/threshold metadata")
    if git("branch","--show-current")!="research-improvements":
        raise ValueError("implementation/validation is restricted to research-improvements")
    folder=build(_sanitize)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:12]
    directory=_safe_output(config.data,run_id)
    attempt_id=uuid.uuid4().hex
    start=time.monotonic();started_at=datetime.now(timezone.utc).isoformat()
    process=None;previous={};received=[];log_bytes=0;ready_read=None;ready_write=None;selector=None
    summary={"run_id":run_id,"attempt_id":attempt_id,"status":"failed","reason":"preflight_failure",
             "retained_records":0,"exit_code":70,"engine_exit_code":None}
    try:
        allocation=register(registry or ROOT/"experiments/pilot/stream_registry.jsonl",
                            seed=config.data["rng"]["seed"],stream=config.data["rng"]["stream"],
                            run_id=run_id,replicate=config.data["replicate"],method=config.data["method"])
        config.data["rng"]["stream"]=allocation["stream"]
        input_bytes=original_bytes if original_bytes is not None else (json.dumps(value,indent=2)+"\n").encode()
        command=_argv(config,folder,directory,_fault_after_bytes)
        manifest={"schema_version":1,"run_id":run_id,"attempt_id":attempt_id,"started_utc":started_at,
                  "input_config_sha256":hashlib.sha256(input_bytes).hexdigest(),
                  "effective_config":config.data,"source":source_identity(),
                  "build":json.loads((folder/"build.json").read_text()),"engine_argv":command,
                  "software":{"python":sys.version,"platform":platform.platform(),
                              "libc":platform.libc_ver(),"numpy":np.__version__},
                  "stream_allocation":allocation,"cap_bound":config.cap,
                  "normalizer_convention":config.ensemble.normalizer_convention,
                  "recording":"uniform proposal schedule including rejected-state holding times",
                  "checkpoint_restart":"unsupported","metadata_storage_reserve":reserve,
                  "numerical_contract":{"uniform":"52-bit midpoint open interval; two PCG32 outputs",
                      "minimum_uniform":2**-53,"maximum_exponential_increment":53*__import__("math").log(2)/config.model.lambda_,
                      "finite_precision":"binary64/libm; integer replay portable, full paths platform dependent",
                      "conditional":"overshoot integrated to infinity; prefix law remains a finite-precision approximation"},
                  "test_injection":bool(_fault_after_bytes or _test_child or _sanitize)}
        preflight_bytes=(len(input_bytes)+len(json.dumps(config.data,indent=2,allow_nan=False).encode())+1+
                         len(json.dumps(manifest,indent=2,allow_nan=False).encode())+1)
        control_reserve=LOG_LIMIT+16384+512*len(config.data["observations"]["thresholds"])
        if preflight_bytes>reserve-control_reserve:
            raise ValueError("manifest exceeds the bounded metadata reserve")
        with (directory/"input-config.json").open("xb") as f:f.write(input_bytes)
        atomic_json(directory/"effective-config.json",config.data)
        atomic_json(directory/"manifest.json",manifest)
        env=os.environ.copy()
        env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
        if _test_child_ready:
            ready_read,ready_write=os.pipe()
            env["TAXI_TEST_READY_FD"]=str(ready_write)
        process=subprocess.Popen(_test_child or command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 cwd=ROOT,env=env,start_new_session=True,
                                 pass_fds=(ready_write,) if ready_write is not None else ())
        if ready_write is not None:
            os.close(ready_write);ready_write=None
        start_ticks=Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()[19]
        atomic_json(directory/"process.json",{"pid":process.pid,"process_group":process.pid,
                    "owner_uid":os.getuid(),"linux_start_ticks":start_ticks,"run_id":run_id,
                    "attempt_id":attempt_id,"scope":"child process group started by this runner"})
        if announce:print(json.dumps({"run_directory":str(directory),"pid":process.pid,"status":"running"}),flush=True)
        def signal_handler(signum,frame):
            received.append(signum)
        for signum in (signal.SIGINT,signal.SIGTERM):
            previous[signum]=signal.signal(signum,signal_handler)
        if _test_child_ready:
            _wait_test_child_ready(process,ready_read,received)
            os.close(ready_read);ready_read=None
        selector=selectors.DefaultSelector()
        selector.register(process.stdout,selectors.EVENT_READ)
        deadline=time.monotonic()+config.data["limits"]["wall_seconds"]
        grace=config.data["limits"]["shutdown_grace_seconds"]
        term_at=None;control_reason=None;peak=0
        with (directory/"engine.log").open("xb") as log:
            while process.poll() is None:
                now=time.monotonic()
                if term_at is None and (received or now>=deadline):
                    control_reason="runner_signal" if received else "supervisor_runtime_limit"
                    os.killpg(process.pid,received[0] if received else signal.SIGTERM)
                    term_at=now
                elif term_at is not None and now-term_at>=grace:
                    os.killpg(process.pid,signal.SIGKILL);control_reason="hard_runtime_timeout";break
                try:
                    for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                        if line.startswith("VmHWM:"):peak=max(peak,int(line.split()[1]))
                except FileNotFoundError:pass
                for key,_ in selector.select(.02):
                    block=os.read(key.fd,65536)
                    if block:
                        remaining=max(0,LOG_LIMIT-log_bytes)
                        log.write(block[:remaining]);log_bytes+=min(len(block),remaining)
                    else:selector.unregister(key.fileobj)
            code=process.wait(timeout=grace+1)
            remainder=process.stdout.read(LOG_LIMIT-log_bytes)
            log.write(remainder)
        summary["engine_exit_code"]=code
        engine_path=directory/"engine-summary.json"
        engine=json.loads(engine_path.read_text()) if engine_path.exists() else None
        summary.update({"status":"completed" if code==0 and engine and engine["status"]=="completed" and not control_reason else
                        ("interrupted" if control_reason or (engine and engine["status"]=="interrupted") else "failed"),
                        "reason":control_reason or (engine["reason"] if engine else "missing_engine_summary"),
                        "exit_code":0 if code==0 and engine and engine["status"]=="completed" and not control_reason else
                                    (124 if control_reason=="hard_runtime_timeout" else (128+received[0] if received else (code if code>0 else 70))),
                        "engine":engine,"peak_rss_kib":max(peak,engine["peak_rss_kib"] if engine else 0)})
        verified=verify_chunks(directory,allow_partial=summary["status"]!="completed")
        summary["retained_records"]=verified["records"]
        summary["unfinalized_files"]=verified["unfinalized_files"]
        atomic_json(directory/"chunks.json",{k:v for k,v in verified.items() if k!="unfinalized_files"})
    except Exception as exc:
        summary.update(status="failed",reason=f"{type(exc).__name__}: {exc}",exit_code=70)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=config.data["limits"]["shutdown_grace_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
        if selector is not None:selector.close()
        if process is not None and process.stdout is not None:process.stdout.close()
        for fd in (ready_read,ready_write):
            if fd is not None:os.close(fd)
        for signum,handler in previous.items():signal.signal(signum,handler)
        summary["wall_seconds_including_supervision"]=time.monotonic()-start
        summary["finished_utc"]=datetime.now(timezone.utc).isoformat()
        summary["written_bytes_before_summary"]=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        summary_bytes=len(json.dumps(summary,indent=2,allow_nan=False).encode())+1
        if summary["written_bytes_before_summary"]+summary_bytes>config.data["limits"]["max_storage_bytes"]:
            summary.update(status="failed",reason="storage_ceiling_violation",exit_code=74)
        atomic_json(directory/"summary.json",summary)
    return directory,summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output-dir")
    parser.add_argument("--stream")
    parser.add_argument("--replicate",type=int)
    args=parser.parse_args()
    config=JobConfig.read(args.config).data
    raw=Path(args.config).read_bytes()
    if args.output_dir:config["output_dir"]=args.output_dir
    if args.stream:config["rng"]["stream"]="auto" if args.stream=="auto" else int(args.stream)
    if args.replicate is not None:config["replicate"]=args.replicate
    directory,summary=run(config,original_bytes=raw,announce=True)
    print(json.dumps({"run_directory":str(directory),"status":summary["status"],
                      "reason":summary["reason"],"retained_records":summary["retained_records"]}))
    return summary["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
