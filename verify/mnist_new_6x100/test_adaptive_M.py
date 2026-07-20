import sys, os, time
sys.path.insert(0, '.')
os.environ.pop('BIGM', None)   # 确保走自适应 M
import cvxpy as cp
import deepmr_mnist_new_6x100_3 as M   # 已改为自适应 M

MODEL = "../../models/mnist_new_6x100/mnist_net_new_6x100.nnet"
PDIR = "../../mnist_properties/mnist_properties_6x100"
# prop 11 应验证成功(1), prop 0 应失败(0) —— 验证自适应 M 与 M=10000 结果一致
for pid, expect in [(11, 1), (0, 0)]:
    net = M.network(); net.load_nnet(MODEL)
    t0 = time.time()
    r = net.find_robustness_number_mrlp(f"{PDIR}/mnist_property_{pid}.txt", 0.019, TRIM=True, WORKERS=28, SOLVER=cp.GUROBI)
    ok = "OK" if r == expect else "MISMATCH!"
    print(f"[adaptM] prop_{pid} verified={r} expect={expect} [{ok}] time={time.time()-t0:.1f}s", flush=True)
print("[adaptM] DONE", flush=True)
