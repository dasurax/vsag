#!/usr/bin/env python3
"""Record hosted compiler memory without changing the compiler resource budget."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

out = Path(sys.argv[1])
command = sys.argv[3:]
if sys.argv[2] != "--" or not command:
    raise SystemExit("usage: probe_factory_memory.py OUTPUT -- COMMAND ...")
out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
samples = []
kernel_pattern = ("out of memory", "killed process", "oom-kill", "oom_reaper")

def counters(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            words = line.replace(":", "").split()
            if len(words) > 1 and words[1].isdigit():
                result[words[0]] = int(words[1])
    except OSError:
        pass
    return result

def snapshot():
    mem = counters("/proc/meminfo")
    vm = counters("/proc/vmstat")
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            if fields.get("Name", "").strip() in ("cc1plus", "cc1"):
                workers.append({"pid": int(entry.name), "rss_kib": int(fields.get("VmRSS", "0").split()[0])})
        except (OSError, ValueError, IndexError):
            pass
    sample = {
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "mem_total_kib": mem.get("MemTotal"),
        "mem_available_kib": mem.get("MemAvailable"),
        "swap_total_kib": mem.get("SwapTotal"),
        "swap_free_kib": mem.get("SwapFree"),
        "compiler_rss_kib": sum(w["rss_kib"] for w in workers),
        "compilers": workers,
        "vmstat": {k: vm.get(k) for k in ("oom_kill", "pswpin", "pswpout", "pgmajfault")},
        "cgroup_memory_events": counters("/sys/fs/cgroup/memory.events"),
    }
    try:
        sample["memory_pressure"] = Path("/proc/pressure/memory").read_text().strip()
    except OSError:
        pass
    samples.append(sample)
    with (out / "memory.jsonl").open("a") as stream:
        stream.write(json.dumps(sample) + "\n")
    print("MEMORY " + json.dumps(sample), flush=True)
    return sample

def kernel_evidence():
    result = subprocess.run(
        ["sudo", "-n", "dmesg", "--ctime"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
    lines = [line for line in result.stdout.splitlines()
             if any(word in line.lower() for word in kernel_pattern)]
    (out / "kernel-oom.log").write_text("\n".join(lines) + "\n" + result.stderr)
    for line in lines[-12:]:
        print("KERNEL " + line, flush=True)

(out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
initial = snapshot()
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    universal_newlines=True, bufsize=1, start_new_session=True)

def relay():
    with (out / "compile.log").open("w") as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)

thread = threading.Thread(target=relay)
thread.start()

def interrupted(signum, frame):
    print("SIGNAL " + str(signum), flush=True)
    snapshot()
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
previous_oom = initial["vmstat"].get("oom_kill")
while process.poll() is None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        current = snapshot()
        current_oom = current["vmstat"].get("oom_kill")
        if current_oom is not None and previous_oom is not None and current_oom > previous_oom:
            kernel_evidence()
        previous_oom = current_oom
thread.join()
snapshot()
kernel_evidence()
summary = {
    "command": command,
    "returncode": process.returncode,
    "elapsed_seconds": round(time.monotonic() - started, 2),
    "peak_compiler_rss_kib": max(s["compiler_rss_kib"] for s in samples),
    "min_mem_available_kib": min(s["mem_available_kib"] for s in samples),
    "min_swap_free_kib": min(s["swap_free_kib"] for s in samples),
    "initial_oom_kill": initial["vmstat"].get("oom_kill"),
    "final_oom_kill": samples[-1]["vmstat"].get("oom_kill"),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("SUMMARY " + json.dumps(summary), flush=True)
raise SystemExit(process.returncode if process.returncode >= 0 else 128 - process.returncode)
