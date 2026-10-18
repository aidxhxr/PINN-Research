"""Resume run_paper_cpu.sh's job list (runs/20261017_paper/cpu_jobs.txt),
skipping fits whose output JSON exists. 4 fits at a time; each fit has a
3-hour wall-clock limit (a hung fit is recorded in resume.log, not retried)."""
import os, re, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "runs/20261017_paper"  # relative to HERE; main() changes into it
env = {**os.environ, "PYTHONPATH": os.path.abspath(os.path.join(HERE, "..", "..", "src"))}


def target(cmd):
    a = shlex.split(cmd)
    out, regime = a[3], a[4]
    start = a[a.index("--start") + 1]
    if "--jitter-seed" in a:
        start += "_j" + a[a.index("--jitter-seed") + 1]
    tag = a[a.index("--tag") + 1] if "--tag" in a else ""
    return os.path.join(out, f"{regime.replace(' ', '_')}_{start}{tag}.json")


def run(cmd):
    t = target(cmd)
    if os.path.exists(t):
        return
    t0 = time.time()
    try:
        subprocess.run(shlex.split(cmd), env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=3 * 3600)
        status = "ok" if os.path.exists(t) else "noout"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    with open(os.path.join(OUT, "resume.log"), "a") as fh:
        fh.write(f"{status} {time.time()-t0:.0f}s {cmd}\n")


def main():
    os.chdir(HERE)
    jobs = [l.strip() for l in open(os.path.join(OUT, "cpu_jobs.txt")) if l.strip()]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if sys.argv[1] in j]
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    todo = [j for j in jobs if not os.path.exists(target(j))]
    print(f"{len(todo)} of {len(jobs)} fits to run", flush=True)
    with ThreadPoolExecutor(npar) as ex:
        list(ex.map(run, todo))
    print("RESUME_DONE", flush=True)


if __name__ == "__main__":
    main()
