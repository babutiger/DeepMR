import os, sys, time
sys.path.insert(0, '.')
import cvxpy as cp
import deepmr_mnist_new_6x100_3_Msweep as MOD

# M 敏感性: 固定网络 + 指定性质集, 扫不同 Big-M 值, 记录每条 verified + 时间
# 结论(机器无关): M 足够大时 V 恒定/结果稳定; M 太小(松弛失效)时结果异常 -> 证明 M=10000 合理
EPS      = float(os.environ.get("MSWEEP_EPS", "0.019"))              # M6=6x100 用 0.019
MODEL    = "../../models/mnist_new_6x100/mnist_net_new_6x100.nnet"   # 团队自训 6x100 (论文 M6, 非 ERAN)
PROPDIR  = "../../mnist_properties/mnist_properties_6x100"
MVALS    = [float(x) for x in os.environ.get("MSWEEP_MVALS", "1,10,100,1000,10000,100000,1000000,10000000").split(",")]
# 核心样本=靠精化(MILP)验证成功的8个; 可用 MSWEEP_PROPS 覆盖
PROPS    = [int(x) for x in os.environ.get("MSWEEP_PROPS", "11,12,49,52,55,58,79,95").split(",")]
WORKERS  = int(os.environ.get("MSWEEP_WORKERS", "28"))
TAG      = os.environ.get("MSWEEP_TAG", "full")

def fmtM(m):
    return str(int(m)) if float(m).is_integer() else str(m)

OUT = f"../../result/original_result/Msweep_mnist6x100_{TAG}.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
f = open(OUT, "w"); f.write("M,property,verified,time_s\n"); f.flush()
print(f"[Msweep] start MVALS={[fmtM(m) for m in MVALS]} PROPS={PROPS} WORKERS={WORKERS} EPS={EPS} -> {OUT}", flush=True)
for M in MVALS:
    os.environ["BIGM"] = fmtM(M)   # verify_lp_split 调用时读取 BIGM
    for pid in PROPS:
        prop = f"{PROPDIR}/mnist_property_{pid}.txt"
        net = MOD.network(); net.load_nnet(MODEL)
        t0 = time.time()
        try:
            res = net.find_robustness_number_mrlp(prop, EPS, TRIM=True, WORKERS=WORKERS, SOLVER=cp.GUROBI)
        except Exception as e:
            res = "ERR"
            print(f"[Msweep] M={fmtM(M)} p={pid} EXCEPTION {repr(e)}", flush=True)
        dt = time.time() - t0
        f.write(f"{fmtM(M)},{pid},{res},{dt:.1f}\n"); f.flush()
        print(f"[Msweep] M={fmtM(M)} p={pid} verified={res} t={dt:.1f}s", flush=True)
f.close()
print("[Msweep] DONE", flush=True)
