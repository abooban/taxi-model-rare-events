"""Content-addressed local C builds; no installation or global configuration."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SANITIZER_FLAGS = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie"]

class BuildError(subprocess.CalledProcessError):
    def __str__(self):
        return (f"command failed (exit {self.returncode}): {shlex.join(self.cmd)}\n"
                f"stdout:\n{self.stdout or ''}\nstderr:\n{self.stderr or ''}")

def _checked_command(command):
    try:
        result = subprocess.run(command, timeout=45, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out: {shlex.join(command)}\n"
                           f"stdout: {exc.stdout!r}\nstderr: {exc.stderr!r}") from exc
    if result.returncode:
        raise BuildError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return result

def _missing_sanitizer_support(error):
    # Only dependency/flag diagnostics from a trivial probe permit a skip.
    # Project compilation errors, probe crashes and timeouts remain failures.
    for line in (error.stderr or "").splitlines():
        runtime = re.search(r"lib(?:asan|ubsan)|-l(?:asan|ubsan)|libclang_rt\.(?:asan|ubsan)", line)
        missing = re.search(r"cannot find|cannot open|No such file or directory|library not found", line)
        unsupported = re.search(r"unrecognized|unsupported|unknown (?:argument|option)", line)
        if (runtime and missing) or (unsupported and "-fsanitize" in line):
            return True
    return False

def sanitizer_unavailable_reason():
    """Return an explicit unsupported-runtime reason; raise on other failures."""
    compiler = os.environ.get("CC", "cc")
    parent = ROOT/"build/validated"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sanitizer-probe.", dir=parent) as temp:
        source = Path(temp)/"probe.c"
        source.write_text("int main(void) { return 0; }\n")
        binary = Path(temp)/"probe"
        command = [compiler, "-std=c11", str(source), "-o", str(binary)]
        _checked_command(command)  # A broken ordinary compiler must never become a skip.
        try:
            _checked_command([*command, *SANITIZER_FLAGS, "-no-pie"])
            _checked_command([str(binary)])
        except BuildError as exc:
            if _missing_sanitizer_support(exc):
                return str(exc)
            raise
    return None

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(sanitize=False):
    compiler = os.environ.get("CC", "cc")
    version = _checked_command([compiler, "--version"]).stdout
    flags = ["-std=c11", "-O1" if sanitize else "-O3", "-g", "-Wall", "-Wextra",
             "-Werror", "-pedantic", "-fno-fast-math"]
    if sanitize:
        flags += SANITIZER_FLAGS
    sources = ["experiments/validated/" + f for f in ("rng.h", "core.h", "core.c")]
    sources += ["tests/validated/test_core.c", "experiments/validated/build.py"]
    has_engine = (ROOT/"experiments/validated/engine.c").exists()
    if has_engine:
        sources += ["experiments/validated/engine.c"]
    source_hashes = {f: digest(ROOT/f) for f in sources}
    identity = {"compiler": compiler, "version": version, "flags": flags, "sources": source_hashes}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    parent = ROOT/"build/validated"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        folder = parent/(("san-" if sanitize else "release-") + key)
        manifest = folder/"build.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if all(digest(folder/p) == h for p,h in result["binaries"].items()):
                return folder
            raise RuntimeError("cached build integrity failure")
        folder.mkdir(exist_ok=True)
        core = str(ROOT/"experiments/validated/core.c")
        commands = []
        outputs = ["core-tests"]
        link_flags = ["-no-pie"] if sanitize else []
        commands.append([compiler, *flags, core, str(ROOT/"tests/validated/test_core.c"),
                         "-lm", *link_flags, "-o", str(folder/"core-tests")])
        if not sanitize:
            outputs += ["libcore.so"]
            commands.append([compiler, *flags, "-shared", "-fPIC", core, "-lm",
                             "-o", str(folder/"libcore.so")])
        if has_engine:
            outputs += ["engine"]
            commands.append([compiler, *flags, core, str(ROOT/"experiments/validated/engine.c"),
                             "-lm", *link_flags, "-o", str(folder/"engine")])
        for command in commands:
            _checked_command(command)
        identity["commands"] = commands
        identity["binaries"] = {p: digest(folder/p) for p in outputs}
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
        return folder

if __name__ == "__main__":
    print(build("--sanitize" in sys.argv))
