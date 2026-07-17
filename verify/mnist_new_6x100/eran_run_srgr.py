import sys, os, time, importlib, signal
sys.path.insert(0, '.')
os.environ.pop('BIGM', None)
import cvxpy as cp

# Generic DeepSRGR driver: use the deepsrgr module (generic class) from the 6x100 directory to run arbitrary nnet+property
# When TIMEOUT>0, apply a per-instance signal timeout -> DNF (to show DeepSRGR is infeasible on the 200 network)
MOD    = importlib.import_module(os.environ["MODULE"])   # e.g. deepsrgr_mnist_new_6x100
MODEL  = os.environ["MODEL"]; PDIR = os.environ["PDIR"]; PREF = os.environ["PREF"]
EPS    = float(os.environ["EPS"]); NPROPS = int(os.environ.get("NPROPS", "100"))
TAG    = os.environ["TAG"]; TIMEOUT = int(os.environ.get("TIMEOUT", "0"))

class TO(Exception): pass
def _h(s, f): raise TO()

OUT = f"../../result/original_result/eran_srgr_{TAG}.csv"
f = open(OUT, "w"); f.write("prop,verified,time_s\n"); f.flush()
print(f"[srgr {TAG}] MODEL={MODEL} EPS={EPS} NPROPS={NPROPS} TIMEOUT={TIMEOUT} -> {OUT}", flush=True)
for i in range(NPROPS):
    prop = f"{PDIR}/{PREF}_{i}.txt"
    net = MOD.network(); net.load_nnet(MODEL)
    t0 = time.time()
    if TIMEOUT > 0:
        signal.signal(signal.SIGALRM, _h); signal.alarm(TIMEOUT)
    try:
        r = net.find_robustness_number_srgrlp(prop, EPS, TRIM=True, WORKERS=28, SOLVER=cp.GUROBI)
    except TO:
        r = "DNF"
    except Exception as e:
        r = "ERR"; print(f"[srgr {TAG}] p={i} EXC {repr(e)[:80]}", flush=True)
    finally:
        if TIMEOUT > 0: signal.alarm(0)
    dt = time.time() - t0
    f.write(f"{i},{r},{dt:.1f}\n"); f.flush()
    print(f"[srgr {TAG}] p={i} v={r} t={dt:.1f}s", flush=True)
    # After DNF, clean up any lingering lpsolve subprocesses
    if r == "DNF":
        os.system("pkill -f 'multiprocessing.*spawn' 2>/dev/null; pkill -f resource_tracker 2>/dev/null")
f.close(); print(f"[srgr {TAG}] DONE", flush=True)
