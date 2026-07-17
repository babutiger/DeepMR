import sys, os, time, importlib
sys.path.insert(0, '.')
# Generic DeepPoly driver (fast): use the deeppoly module in the 6x100 directory to run any nnet + property
MOD    = importlib.import_module(os.environ["MODULE"])   # deeppoly_mnist_new_6x100
MODEL  = os.environ["MODEL"]; PDIR = os.environ["PDIR"]; PREF = os.environ["PREF"]
EPS    = float(os.environ["EPS"]); NPROPS = int(os.environ.get("NPROPS", "100")); TAG = os.environ["TAG"]
OUT = f"../../result/original_result/eran_poly_{TAG}.csv"
f = open(OUT, "w"); f.write("prop,verified,time_s\n"); f.flush()
print(f"[poly {TAG}] MODEL={MODEL} EPS={EPS} NPROPS={NPROPS} -> {OUT}", flush=True)
for i in range(NPROPS):
    net = MOD.network(); net.load_nnet(MODEL)
    t0 = time.time()
    try:
        r = net.find_robustness_number_poly(f"{PDIR}/{PREF}_{i}.txt", EPS, TRIM=True)
    except Exception as e:
        r = "ERR"; print(f"[poly {TAG}] p={i} EXC {repr(e)[:80]}", flush=True)
    dt = time.time() - t0
    f.write(f"{i},{r},{dt:.1f}\n"); f.flush()
f.close(); print(f"[poly {TAG}] DONE V={sum(1 for _ in open(OUT) if _.strip().endswith(',1') or ',1,' in _)}", flush=True)
