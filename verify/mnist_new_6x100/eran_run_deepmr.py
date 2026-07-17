import sys, os, time, importlib, signal
sys.path.insert(0, '.')
# ERAN DeepMR: fixed M=10000 (consistent with the paper's other tables); when TIMEOUT>0, per-instance timeout -> DNF (for time control on 200 networks)
os.environ['BIGM'] = os.environ.get('BIGM', '10000')
import cvxpy as cp
MOD    = importlib.import_module(os.environ["MODULE"])
MODEL  = os.environ["MODEL"]; PDIR = os.environ["PDIR"]; PREF = os.environ["PREF"]
EPS    = float(os.environ["EPS"]); NPROPS = int(os.environ.get("NPROPS", "100")); TAG = os.environ["TAG"]
TIMEOUT = int(os.environ.get("TIMEOUT", "0"))
class TO(Exception): pass
def _h(s, fr): raise TO()
OUT = f"../../result/original_result/eran_deepmr_{TAG}.csv"
f = open(OUT, "w"); f.write("prop,verified,time_s\n"); f.flush()
print(f"[deepmr {TAG}] MODEL={MODEL} EPS={EPS} M={os.environ['BIGM']} NPROPS={NPROPS} TIMEOUT={TIMEOUT} -> {OUT}", flush=True)
for i in range(NPROPS):
    net = MOD.network(); net.load_nnet(MODEL)
    t0 = time.time()
    if TIMEOUT > 0:
        signal.signal(signal.SIGALRM, _h); signal.alarm(TIMEOUT)
    try:
        r = net.find_robustness_number_mrlp(f"{PDIR}/{PREF}_{i}.txt", EPS, TRIM=True, WORKERS=28, SOLVER=cp.GUROBI)
    except TO:
        r = "DNF"
    except Exception as e:
        r = "ERR"; print(f"[deepmr {TAG}] p={i} EXC {repr(e)[:80]}", flush=True)
    finally:
        if TIMEOUT > 0: signal.alarm(0)
    dt = time.time() - t0
    f.write(f"{i},{r},{dt:.1f}\n"); f.flush()
    print(f"[deepmr {TAG}] p={i} v={r} t={dt:.1f}s", flush=True)
    if r == "DNF":
        os.system("pkill -f 'multiprocessing.*spawn' 2>/dev/null")
f.close(); print(f"[deepmr {TAG}] DONE", flush=True)
