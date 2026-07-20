import sys, os, time, importlib
sys.path.insert(0, '.')
os.environ.pop('BIGM', None)   # 走自适应 M(有效Big-M)
import cvxpy as cp

# 用自适应 M 重跑一个网络的全部实例, 与原 M=10000 结果对比(核查 M9/M10 假阳性)
MOD    = importlib.import_module(os.environ["DEEPMR_MOD"])
MODEL  = os.environ["MODEL"]; PDIR = os.environ["PDIR"]; PREF = os.environ["PREF"]
EPS    = float(os.environ["EPS"]); NPROPS = int(os.environ.get("NPROPS", "100")); TAG = os.environ["TAG"]
OUT = f"../../result/original_result/rerun_{TAG}.csv"
f = open(OUT, "w"); f.write("prop,verified,time_s\n"); f.flush()
print(f"[rerun {TAG}] MODEL={MODEL} EPS={EPS} NPROPS={NPROPS} -> {OUT}", flush=True)
for i in range(NPROPS):
    net = MOD.network(); net.load_nnet(MODEL)
    t0 = time.time()
    try:
        r = net.find_robustness_number_mrlp(f"{PDIR}/{PREF}_{i}.txt", EPS, TRIM=True, WORKERS=28, SOLVER=cp.GUROBI)
    except Exception as e:
        r = "ERR"; print(f"[rerun {TAG}] p={i} EXC {repr(e)[:80]}", flush=True)
    dt = time.time() - t0
    f.write(f"{i},{r},{dt:.1f}\n"); f.flush()
    print(f"[rerun {TAG}] p={i} v={r} t={dt:.1f}s", flush=True)
f.close(); print(f"[rerun {TAG}] DONE", flush=True)
