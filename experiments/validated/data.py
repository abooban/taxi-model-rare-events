"""TAXIv1 little-endian chunk parser and integrity verifier.

Header: magic[8], kind:u32, threshold_count:u32, width:u64, reserved:u64.
Sample record (48 bytes): index:u64,C:f64,X:f64,N:u64,long_steps:u64,flags:u64.
Conditional record: index:u64,K:u64,pre:f64,a:f64,conditional_mean_C:f64,
then threshold_count log survival probabilities:f64. CRC32 covers the header too.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import zlib
import numpy as np
from .jobconfig import JobConfig

SAMPLE_DTYPE=np.dtype([("index","<u8"),("C","<f8"),("X","<f8"),("N","<u8"),
                       ("long_steps","<u8"),("flags","<u8")])

class IntegrityError(ValueError):
    pass

def conditional_dtype(count):
    return np.dtype([("index","<u8"),("K","<u8"),("pre","<f8"),("a","<f8"),
                     ("mean_C","<f8"),("log_tails","<f8",(count,))])

def _load(path):
    return json.loads(Path(path).read_text())

def verify_chunks(directory, *, allow_partial=False, collect=False, max_memory_bytes=512*1024**2):
    root=Path(directory)
    manifest=_load(root/"manifest.json")
    config=JobConfig(manifest["effective_config"])
    d=config.data; obs=d["observations"]
    conditional=d["method"].startswith("conditional")
    kind=2 if conditional else 1
    dtype=conditional_dtype(len(obs["thresholds"])) if conditional else SAMPLE_DTYPE
    partials=sorted(p.name for p in root.glob("chunk-*") if p.name.endswith((".part",".tmp")))
    metas=sorted(root.glob("chunk-*.bin.meta.json"))
    binaries=sorted(root.glob("chunk-*.bin"))
    if len(metas)!=len(binaries):
        partials+=sorted(p.name for p in binaries if not Path(str(p)+".meta.json").exists())
    if partials and not allow_partial:raise IntegrityError(f"unfinalized/truncated output: {partials}")
    known={}
    if (root/"chunks.json").exists():
        known={x["file"]:x for x in _load(root/"chunks.json")["chunks"]}
        if set(known)!={p.name.removesuffix(".meta.json") for p in metas}:
            raise IntegrityError("published chunk index differs from actual files")
    count=0;total=0;chunks=[];arrays=[]
    for sequence,path in enumerate(metas):
        meta=_load(path)
        name=meta.get("file","")
        if (name!=path.name.removesuffix(".meta.json") or
            not re.fullmatch(r"chunk-[0-9]{8,}\.bin",name) or
            name!=f"chunk-{sequence:08d}.bin"):
            raise IntegrityError("invalid or missing chunk sequence")
        binary=root/name
        if binary.is_symlink() or not binary.is_file():raise IntegrityError("missing/symlink chunk")
        size=binary.stat().st_size
        if size!=meta["bytes"] or size!=32+meta["records"]*dtype.itemsize or meta["records"]<1:
            raise IntegrityError("truncated chunk or wrong actual record count")
        if collect and total+size>max_memory_bytes:raise MemoryError("analysis memory ceiling; use streaming or a declared recording stride")
        crc=0;sha=hashlib.sha256()
        with binary.open("rb") as f:
            header=f.read(32)
            if len(header)!=32 or struct.unpack("<8sIIQQ",header)!=(b"TAXIv1\0\0",kind,len(obs["thresholds"]),dtype.itemsize,0):
                raise IntegrityError("unknown binary header/schema")
            f.seek(0)
            for block in iter(lambda:f.read(1024*1024),b""):
                crc=zlib.crc32(block,crc);sha.update(block)
        if f"{crc:08x}"!=meta["crc32"]:raise IntegrityError("CRC32 mismatch")
        hexdigest=sha.hexdigest()
        if known and known[name]["sha256"]!=hexdigest:raise IntegrityError("SHA256 mismatch")
        a=np.memmap(binary,dtype=dtype,mode="r",offset=32,shape=(meta["records"],))
        expected=d["warmup"]+1+(count+np.arange(len(a),dtype=np.uint64))*obs["stride"]
        if not np.array_equal(a["index"],expected):raise IntegrityError("missing/duplicated/incorrect proposal indices")
        if int(a["index"][0])!=meta["first_index"] or int(a["index"][-1])!=meta["last_index"]:
            raise IntegrityError("chunk index metadata mismatch")
        if conditional:
            if (np.any(~np.isfinite(a["pre"])) or np.any(~np.isfinite(a["a"])) or
                np.any(~np.isfinite(a["mean_C"])) or np.any(a["a"]<=0) or np.any(a["a"]>config.model.L) or
                np.any(a["pre"]<a["K"]) or np.any(a["mean_C"]<a["K"]+1) or
                np.any(np.isnan(a["log_tails"])) or np.any(a["log_tails"]>0) or
                np.any(a["log_tails"][:,1:]>a["log_tails"][:,:-1])):
                raise IntegrityError("invalid conditional records")
        else:
            if (np.any(~np.isfinite(a["C"])) or np.any(~np.isfinite(a["X"])) or
                np.any(a["N"]<1) or np.any(a["N"]>2**53) or np.any(a["C"]<a["N"]) or
                np.any(a["X"]<config.model.L) or np.any(a["long_steps"]>a["N"]) or
                np.any(a["flags"]>3)):
                raise IntegrityError("invalid physical observations")
            atoms=(a["long_steps"]==0) if config.model.b>0 else np.ones(len(a),bool)
            if np.any(a["C"][atoms]!=a["N"][atoms]):raise IntegrityError("invalid structural atoms")
            if config.model.b>0 and np.any(a["N"][atoms]*config.model.eta_c<=config.model.L):
                raise IntegrityError("impossible atom")
            if config.ensemble.kind=="finite_M" and np.any(a["N"]>config.ensemble.max_steps):
                raise IntegrityError("finite-M support violation")
        item={**meta,"sha256":hexdigest}
        chunks.append(item);count+=len(a);total+=size
        if collect:arrays.append(np.array(a))
        del a
    engine_path=root/"engine-summary.json"
    if engine_path.exists():
        engine=_load(engine_path)
        if engine["retained_records"]!=count or engine["finalized_chunks"]!=len(chunks):
            raise IntegrityError("engine/chunk retained counts disagree")
        if engine["status"]=="completed":
            expected_count=(d["iterations"]-1)//obs["stride"]+1
            if count!=expected_count or engine["iterations_completed"]!=d["iterations"]+d["warmup"]:
                raise IntegrityError("false completion or incorrect retained observation count")
    elif not allow_partial:raise IntegrityError("missing engine completion summary")
    summary_path=root/"summary.json"
    if summary_path.exists() and not allow_partial:
        summary=_load(summary_path)
        if summary["status"]!="completed":raise IntegrityError("run is interrupted/failed; explicit partial audit required")
        if summary["retained_records"]!=count:raise IntegrityError("runner retained count mismatch")
    return {"records":count,"binary_bytes":total,"chunks":chunks,"unfinalized_files":partials,
            "array":np.concatenate(arrays) if arrays else np.empty(0,dtype=dtype)} if collect else {
                "records":count,"binary_bytes":total,"chunks":chunks,"unfinalized_files":partials}

def read_observations(directory, *, max_memory_bytes=512*1024**2):
    return verify_chunks(directory,collect=True,max_memory_bytes=max_memory_bytes)["array"]
