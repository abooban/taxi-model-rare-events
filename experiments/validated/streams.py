"""Locked append-only allocation. Unique selectors are not a proof of independence."""
from pathlib import Path
import fcntl
import json
import os

def integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value

def register(path, *, seed, stream, run_id, replicate, method):
    integer("seed",seed,0,2**64-1)
    integer("replicate",replicate,0,2**31-1)
    if stream != "auto": integer("stream",stream,0,2**63-1)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        handle.seek(0)
        data=handle.read(16*1024*1024+1)
        if len(data)>16*1024*1024 or (data and not data.endswith("\n")):
            raise ValueError("stream registry oversized or truncated; manual audit required")
        rows=[json.loads(line) for line in data.splitlines()]
        used={row["stream"] for row in rows}
        if len(used)!=len(rows):raise ValueError("duplicate selector in registry")
        if stream == "auto": stream=max({999999}|used)+1
        integer("stream",stream,0,2**63-1)
        if stream in used:raise ValueError(f"stream {stream} already registered; independent jobs cannot reuse it")
        row={"family":"pcg32-xsh-rr-64-32","seed":seed,"stream":stream,
             "run_id":run_id,"replicate":replicate,"method":method}
        handle.seek(0,os.SEEK_END)
        handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        handle.flush();os.fsync(handle.fileno())
    return row
